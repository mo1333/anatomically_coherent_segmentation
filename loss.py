from __future__ import annotations

from typing import Tuple, Any

import torch
from monai.losses import DiceCELoss
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss


class TotalLoss(_Loss):
    def __init__(self,
                 config,
                 device,
                 reduction: LossReduction | str = LossReduction.MEAN):
        super().__init__(reduction=LossReduction(reduction).value)
        loss_config = config["loss_config"]
        self.device = device
        self.diceCeLoss = DiceCELoss(sigmoid=bool(loss_config["sigmoid"]),
                                     softmax=bool(loss_config["softmax"]),
                                     lambda_dice=loss_config["lambda_dice"],
                                     lambda_ce=loss_config["lambda_ce"],
                                     include_background=bool(loss_config["include_background"]))

        # self.topologyLoss = NaiveTopologyLoss(sigmoid=bool(loss_config["sigmoid"]),
        #                                       softmax=bool(loss_config["softmax"]),
        #                                       post_func=loss_config["post_func"])

        if loss_config["lambda_top"] >= 1e-8:
            self.topologyLoss = TopologyLoss(sigmoid=bool(loss_config["sigmoid"]),
                                             softmax=bool(loss_config["softmax"]))
        else:
            self.topologyLoss = DummyLoss(device=device)

        if loss_config["lambda_cdr"] >= 1e-8:
            self.cdrloss = CDRLoss(sigmoid=bool(loss_config["sigmoid"]),
                                   softmax=bool(loss_config["softmax"]),
                                   device=device)
        else:
            self.cdrloss = DummyLoss(device=device)

        self.loss_config = loss_config

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> tuple[Any, Any, Any, Any]:
        return (self.diceCeLoss(input, target) +
                self.loss_config["lambda_top"] * self.topologyLoss(input, target) +
                self.loss_config["lambda_cdr"] * self.cdrloss(input, target),
                self.diceCeLoss(input, target),
                self.loss_config["lambda_top"] * self.topologyLoss(input, target),
                self.loss_config["lambda_cdr"] * self.cdrloss(input, target))


class DummyLoss(_Loss):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        zero_mask = torch.zeros_like(input).to(self.device)
        return torch.mean(input * zero_mask)


class NaiveTopologyLoss(_Loss):
    def __init__(self, sigmoid, softmax, post_func):
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.post_func = post_func

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param input:
        :return:

        required input shape: BCHW(D)
        """

        func_dict = {"": None,
                     "square": torch.square,
                     "exp": lambda x: torch.sub(torch.exp(x), 1),
                     "log": lambda x: torch.log(torch.add(x, 1))}

        if self.sigmoid:
            y = torch.sigmoid(input)
        if self.softmax:
            y = torch.softmax(input, dim=1)

        diff = y[:, 1] - y[:, 2]
        f = torch.clamp(diff, min=0)  # apply elementwise max(x, 0) to diff
        if func_dict[self.post_func]:
            f = func_dict[self.post_func](f)
        return torch.mean(f)


class TopologyLoss(_Loss):
    def __init__(self, sigmoid, softmax):
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param target:
        :param input:
        :return:

        required input shape: BCHW(D)
        """

        if self.sigmoid:
            y = torch.sigmoid(input)
        if self.softmax:
            y = torch.softmax(input, dim=1)

        log_y = torch.log(y + 1e-8)  # needs small, to prevent taking log(0)
        prod = - torch.mul(target, log_y)
        sum_over_channels = torch.sum(prod, dim=1)

        # Assumption: a prediction is valid according to topology, when the prob. for cup is smaller than disc
        V = y[:, 1] <= y[:, 2]

        return torch.mean(torch.mul(sum_over_channels, V))


class CDRLoss(_Loss):
    def __init__(self, sigmoid, softmax, device, offset=0.05, strech_factor=1/2):
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.device = device
        self.offset = torch.tensor(offset).to(self.device)
        self.strech_factor = strech_factor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param target:
        :param input:
        :return:

        required input shape: BCHW(D)
        with channels being 0... background, 1... optic cup, 2... optic disc
        """

        if self.sigmoid:
            y = torch.sigmoid(input)
        if self.softmax:
            y = torch.softmax(input, dim=1)

        label_cup_diameter = get_vertical_diameter(target[:, 1])
        label_disc_diameter = get_vertical_diameter(target[:, 2])
        pred_cup_diameter = get_vertical_diameter(y[:, 1] >= 0.5)
        pred_disc_diameter = get_vertical_diameter(y[:, 2] >= 0.5)

        # the ratio should not be able to exceed 1, therefore clamp with max=1
        mse = torch.square(
            torch.div(label_cup_diameter, label_disc_diameter) - torch.clamp(
                torch.div(pred_cup_diameter, pred_disc_diameter), max=1)).to(
            self.device)
        mask = torch.tensor([0, 1, 0]).to(self.device)  # we only want the cup to change
        mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        y_masked = y * mask

        mse = mse.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        return torch.mean((- self.strech_factor * torch.square(y_masked) + y_masked + self.offset) * mse)


def get_vertical_diameter(images):
    """
    :param images:
    :return:

    required input shape: BHW
    """
    batch_size = images.shape[0]

    # returns a 3-tuple, containing all coordinates of active points
    indices = torch.where(images >= 0.5)

    # get all indices of the variable "indices" which contain each batch index
    indices_per_id = [torch.where(indices[0] == i) for i in range(batch_size)]

    try:
        # subtract first index of appearance in H dimension from last one to get vertical pixel-diameter
        diameters = torch.tensor(
            [(indices[1][indices_per_id[i][0][-1]] - indices[1][indices_per_id[i][0][0]]).item() for i in
             range(batch_size)])
    except IndexError:
        # Hotfix: When one of the images in the batch has not predicted disc, or cup pixels,
        # the loss for this batch is set to 0 (in order for the dice loss to be able to repair)
        # Maybe change this later
        diameters = torch.tensor([1] * batch_size)
    return diameters
