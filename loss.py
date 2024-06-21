from __future__ import annotations

from typing import Tuple, Any

import torch
from monai.losses import DiceCELoss
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss


class TotalLoss(_Loss):
    def __init__(self,
                 loss_config,
                 reduction: LossReduction | str = LossReduction.MEAN):
        super().__init__(reduction=LossReduction(reduction).value)
        self.diceCeLoss = DiceCELoss(sigmoid=bool(loss_config["sigmoid"]),
                                     softmax=bool(loss_config["softmax"]),
                                     lambda_dice=loss_config["lambda_dice"],
                                     lambda_ce=loss_config["lambda_ce"],
                                     include_background=bool(loss_config["include_background"]))

        # self.topologyLoss = NaiveTopologyLoss(sigmoid=bool(loss_config["sigmoid"]),
        #                                       softmax=bool(loss_config["softmax"]),
        #                                       post_func=loss_config["post_func"])

        self.topologyLoss = TopologyLoss(sigmoid=bool(loss_config["sigmoid"]),
                                         softmax=bool(loss_config["softmax"]))

        self.loss_config = loss_config

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> tuple[Any, Any, Any]:
        return (self.diceCeLoss(input, target) + self.loss_config["lambda_top"] * self.topologyLoss(input, target),
                self.diceCeLoss(input, target),
                self.loss_config["lambda_top"] * self.topologyLoss(input, target))


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

        log_y = torch.log(y + 1e-8) # sometimes training fail. Maybe due to taking log?
        prod = - torch.mul(target, log_y)
        sum_over_channels = torch.sum(prod, dim=1)

        # Assumption: a prediction is valid according to topology, when the prob. for cup is smaller than disc
        V = (y[:, 1] <= y[:, 2]).int()

        return torch.mean(torch.mul(sum_over_channels, V))

class CDRLoss(_Loss):
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

