# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weight_reduce_loss
import copy
import numpy as np


@MODELS.register_module()
class AFMALoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='loss_afma',
                 att_depth=2, out_channels=4, patch_size=16
                 ):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int | None): The label index to be ignored.
                Default: 255.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_dice'.
        """

        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self._loss_name = loss_name

        self.mseloss = nn.MSELoss()
        self.att_depth = att_depth
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                stride=(self.patch_size, self.patch_size))
        self.activation = nn.Softmax(dim=1)

    def forward(self,
                pred,
                target,
                attentions,
                weight=None,
                avg_factor=None,
                reduction_override=None
                ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = self.activation(pred)
            else:
                raise NotImplementedError

        new_target = target
        label_list = []
        for batch in range(pred.shape[0]):
            list = []
            for i in range(0, pred.shape[1]):
                label = copy.deepcopy(new_target[batch])
                label[label != i] = -1
                label[label == i] = 1
                label[label == -1] = 0
                list.append(label)
            list = torch.stack(list, dim=0)
            label_list.append(list)
        labels = torch.stack(label_list, dim=0)

        y_gt = labels.to(dtype=torch.float32)
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels,
                                     kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                                     stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels,
                                     bias=False)
        conv_feamap_size.weight = nn.Parameter(
            torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(pred.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        y_gt_conv = conv_feamap_size(y_gt) / (2 ** self.att_depth * 2 ** self.att_depth)

        attentions_gt = []

        for i in range(y_gt_conv.size()[1]):
            unfold_y_gt = self.unfold(y_gt[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_y_gt_conv = self.unfold(y_gt_conv[:, i:i + 1, :, :])
            att = torch.matmul(unfold_y_gt, unfold_y_gt_conv) / (self.patch_size * self.patch_size)
            att = torch.unsqueeze(att, dim=1)
            attentions_gt.append(att)

        attentions_gt = torch.cat(attentions_gt, dim=1)

        # y_gt = torch.argmax(y_gt, dim=-3)

        loss_mse = self.mseloss(attentions, attentions_gt)

        return loss_mse

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
