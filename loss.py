from __future__ import annotations

from typing import Tuple, Any

import torch
from monai.losses import DiceCELoss, DiceLoss
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
        self.diff_rounding = DifferentiableRounding()

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


        # we penalize cup everwhere we do not have a disc prob > 0.5
        disc = self.diff_rounding(y[:, 2]) # add eps to change threshold from 0.5
        diff = y[:, 1] - disc

        f = torch.clamp(diff, min=0)  # apply elementwise max(x, 0) to diff
        if func_dict[self.post_func]:
            f = func_dict[self.post_func](f)
        return torch.mean(f)


class DifferentiableRounding(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


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

        log_y = torch.log(y + 1e-8)  # needs small eps, to prevent taking log(0)
        prod = - torch.mul(target, log_y)
        sum_over_channels = torch.sum(prod, dim=1)

        # Assumption: a prediction is valid according to topology, when the prob. for cup is smaller than disc
        V = y[:, 1] <= y[:, 2]

        return torch.mean(torch.mul(sum_over_channels, V))


class CDRLoss(_Loss):
    def __init__(self, sigmoid, softmax, device, offset=0.05, stretch_factor=1 / 2):
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.device = device
        self.offset = torch.tensor(offset).to(self.device)
        self.stretch_factor = stretch_factor

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

        true_ratio = torch.div(label_cup_diameter, label_disc_diameter)

        # the ratio should not be able to exceed 1, therefore clamp with max=1
        pred_ratio = torch.clamp(torch.div(pred_cup_diameter, pred_disc_diameter), max=1)
        mse = torch.square(true_ratio - pred_ratio).to(self.device)
        mask = torch.tensor([0, 1, 0]).to(self.device)  # we only want the cup to change
        mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(2)

        y_masked = y * mask

        # if the predicted ratio is smaller than the ground truth, we have to enhance the optic cup,
        # which can be done by switching the sign (but only at those elements in the batch, where this is the case)
        y_masked[pred_ratio < true_ratio] = 1 - (y * mask)[pred_ratio < true_ratio]

        mse = mse.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        return torch.mean((- self.stretch_factor * torch.square(y_masked) + y_masked + self.offset) * mse)


def get_vertical_diameter(images):
    """
    :param images:
    :return:

    required input shape: BHW
    """
    batch_size = images.shape[0]

    # returns a 3-tuple, containing all coordinates of active points
    indices = torch.where(images)

    # get all indices of the variable "indices" which contain each batch index
    indices_per_id = [torch.where(indices[0] == i) for i in range(batch_size)]

    try:
        # subtract first index of appearance in H dimension from last one to get vertical pixel-diameter
        diameters = torch.tensor(
            [(indices[1][indices_per_id[i][0][-1]] - indices[1][indices_per_id[i][0][0]]).item() for i in
             range(batch_size)])
    except IndexError:
        # Hotfix: When one of the images in the batch has not predicted disc, or cup pixels,
        # the 1 (0 does not work, since we compute ratios)
        # Maybe change this later
        diameters = torch.tensor([1] * batch_size)
    return diameters


class TopUNetLoss(_Loss):
    def __init__(self, loss_config):
        super(TopUNetLoss, self).__init__()
        self.loss_config = loss_config
        self.dice = DiceCELoss(lambda_dice=loss_config["lambda_dice"],
                               lambda_ce=loss_config["lambda_ce"],
                               include_background=bool(loss_config["include_background"]))
        self.kldiv = torch.nn.KLDivLoss()
        self.l1 = torch.nn.L1Loss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> tuple[Any, Any, Any, Any]:
        return (self.dice(input[0], target[0]) +
                self.loss_config["lambda_kl"] * self.kldiv(input[1], target[1]) +
                self.loss_config["lambda_l1"] * self.l1(input[2], target[2]),
                self.dice(input[0], target[0]),
                self.loss_config["lambda_kl"] * self.kldiv(input[1], target[1]),
                self.loss_config["lambda_l1"] * self.l1(input[2], target[2]))
