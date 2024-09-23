# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weight_reduce_loss
import torch.nn.functional as F
import numpy as np

def dice_loss(
    pred,
    target,
    weight,
    eps=1e-3,
    reduction='mean',
    naive_dice=False,
    avg_factor=None,
):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
            loss defined in the V-Net paper, otherwise, use the
            naive dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def tversky(pred,
                 target,
                 valid_mask,
                 alpha=0.3,
                 beta=0.7,
                 smooth=1,
                 class_weight=None,
                 ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    one_hot_target = F.one_hot(
        torch.clamp(target.long(), 0, num_classes - 1),
        num_classes=num_classes)
    total_loss = 0
    for i in range(num_classes):
        if i != ignore_index:
            tversky_loss = binary_tversky_loss(
                pred[:, i],
                one_hot_target[..., i],
                valid_mask=valid_mask,
                alpha=alpha,
                beta=beta,
                smooth=smooth)
            if class_weight is not None:
                tversky_loss *= class_weight[i]
            total_loss += tversky_loss
    return total_loss / num_classes



def binary_tversky_loss(pred,
                        target,
                        valid_mask,
                        alpha=0.3,
                        beta=0.7,
                        smooth=1):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    TP = torch.sum(torch.mul(pred, target) * valid_mask, dim=1)
    FP = torch.sum(torch.mul(pred, 1 - target) * valid_mask, dim=1)
    FN = torch.sum(torch.mul(1 - pred, target) * valid_mask, dim=1)
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return tversky


@MODELS.register_module()
class My_DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 activate=False,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='loss_dice'):
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

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
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
        ignore = True

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            # else:
            #     raise NotImplementedError

        new_target = target.clone()
        label_list = []
        for batch in range(pred.shape[0]):
            list = []
            for i in range(pred.shape[1]):
                label = new_target[batch].clone()
                label[label != i] = -1
                label[label == i] = 1
                label[label == -1] = 0
                list.append(label)
            list = torch.stack(list, dim=0)
            label_list.append(list)
        labels = torch.stack(label_list, dim=0)

        for batch in range(pred.shape[0]):
            batch_loss = 0
            for i in range(pred.shape[1]):
                if ignore == True and i == 0:
                    pass
                else:
                    loss = self.loss_weight * dice_loss(
                        pred[batch][i],
                        labels[batch][i],
                        weight,
                        eps=self.eps,
                        reduction=reduction,
                        naive_dice=self.naive_dice,
                        avg_factor=avg_factor,
                    )
                    batch_loss = batch_loss + loss

        total_loss = batch_loss

        return total_loss

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
