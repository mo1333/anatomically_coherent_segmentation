from __future__ import annotations

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

        self.topologyLoss = NaiveTopologyLoss(sigmoid=bool(loss_config["sigmoid"]),
                                              softmax=bool(loss_config["softmax"]))
        self.loss_config = loss_config

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.diceCeLoss(input, target) + self.loss_config["lambda_top"] * self.topologyLoss(input)


class NaiveTopologyLoss(_Loss):
    def __init__(self, sigmoid, softmax):
        super().__init__()
        self.sigmoid = sigmoid
        self.softmax = softmax

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        :param input:
        :return:

        required input shape: BCWH(D)
        """
        if self.sigmoid:
            y = torch.sigmoid(input)
        if self.softmax:
            y = torch.softmax(input, dim=1)
        diff = y[:, 1] - y[:, 2]
        f = torch.clamp(diff, min=0)  # apply elementwise max(x, 0) to diff
        return torch.mean(f)
