# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

from ..utils import resize
from torch import Tensor
from typing import List, Tuple
from mmseg.utils import ConfigType, SampleList

import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
from ..losses import accuracy
from ..losses.my_dice_loss import tversky


@MODELS.register_module()
class My_FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size

        super().__init__(**kwargs)

        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)



    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)

        feats = inputs

        return feats

    def forward(self, inputs):
        """Forward function."""
        # output = self._forward_feature(inputs)
        # output = self.cls_seg(output)
        return inputs

    def loss(self, inputs, batch_data_samples: SampleList) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)

        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        # seg_nc_logits = inputs[0]
        # seg_hv_logits = inputs[1]

        # h_grads, v_grads = self.get_gradient_hv(seg_hv_logits, h_ch=1, v_ch=0)

        # nc_losses = self.loss_by_feat(seg_nc_logits, batch_data_samples)
        # hv_losses = 0
        # for i in range(0, seg_hv_logits.shape[0]):
        # hv_losses = F.mse_loss(seg_hv_logits[:, 1:2, :, :], h_grads) + F.mse_loss(seg_hv_logits[:, 0:1, :, :], v_grads)

        # nc_losses['loss_hv'] = hv_losses
        # losses = nc_losses

        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        # seg_logits = self.forward(inputs[0])
        seg_logits = self.forward(inputs)

        return self.predict_by_feat(seg_logits, batch_img_metas)

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        seg_logits = resize(
            input=seg_logits,
            # size=batch_img_metas[0]['img_shape'],
            size=batch_img_metas[0].img_shape,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        total_dice = 0
        dice = tversky(
                seg_logits, seg_label, valid_mask=(seg_label != self.ignore_index).long(), alpha=0.5, beta=0.5)
        for i in range(seg_logits.shape[0]):
            total_dice = total_dice +dice[i]
        loss['seg_dice'] = total_dice/(seg_logits.shape[0] - 1) * 100
        return loss


    def get_sobel_filters(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get horizontal and vertical sobel filters

        Parameters
        ----------
        size : int
            Size of sobel filter

        Returns
        -------
        kernels : Tuple[torch.Tensor, torch.Tensor]
            Horizontal & vertical sobel filters
        """
        assert size % 2 == 1, "Size must be odd"
        device = torch.device('cuda:0')
        h_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=device)
        v_range = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=device)
        h, v = torch.meshgrid([h_range, v_range])
        h, v = h.transpose(0, 1), v.transpose(0, 1)

        kernel_h = h / (h * h + v * v + 1e-15)
        kernel_v = v / (h * h + v * v + 1e-15)

        return kernel_h, kernel_v

    def get_gradient_hv(self, logits: torch.Tensor,
                        h_ch: int = 1,
                        v_ch: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get horizontal & vertical gradients

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits from HV branch
        h_ch : int
            Number of horizontal channels
        v_ch : int
            Number of vertical channels

        Returns
        -------
        gradients : Tuple[torch.Tensor, torch.Tensor]
            Horizontal and vertical gradients
        """
        mh, mv = self.get_sobel_filters(size=5)
        mh = mh.reshape(shape=(1, 1, 5, 5))
        mv = mv.reshape(shape=(1, 1, 5, 5))

        # hl = logits[..., h_ch].unsqueeze(dim=-1)
        # vl = logits[..., v_ch].unsqueeze(dim=-1)
        hl = logits[:, h_ch:h_ch+1, :, :]
        vl = logits[:, v_ch:v_ch+1, :, :]

        dh = F.conv2d(hl, mh, stride=1, padding=2)
        dv = F.conv2d(vl, mv, stride=1, padding=2)

        # sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)


        return dh, dv



class DiceCoeff(nn.Module):
    # See: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor,
                smooth: float = 1e-7) -> torch.Tensor:
        if not (torch.max(inputs) == 1 and torch.min(inputs) >= 0):
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        iflat = probs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return ((2.0 * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class _NPBranchLoss(nn.Module):
    def __init__(self):
        super(_NPBranchLoss, self).__init__()
        self.dice_coeff = DiceCoeff()

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        loss = (F.cross_entropy(logits, targets) +
                1 - self.dice_coeff(logits, targets))
        return loss


class _HVBranchLoss(nn.Module):
    def __init__(self):
        super(_HVBranchLoss, self).__init__()

    def forward(self,
                logits: torch.Tensor,
                h_grads: torch.Tensor,
                v_grads: torch.Tensor) -> torch.Tensor:
        # MSE of vertical and horizontal gradients with logits
        loss = F.mse_loss(logits, h_grads) + F.mse_loss(logits, v_grads)
        return loss


class HoverLoss(nn.Module):
    def __init__(self):
        super(HoverLoss, self).__init__()
        self.np_loss = _NPBranchLoss()
        self.hv_loss = _HVBranchLoss()

    def forward(self, np_logits, np_targets,
                hv_logits, h_grads, v_grads,
                weights=(1, 1)) -> torch.Tensor:
        loss = (self.np_loss(np_logits, np_targets) * weights[0] +
                self.hv_loss(hv_logits, h_grads, v_grads) * weights[1])
        return loss



