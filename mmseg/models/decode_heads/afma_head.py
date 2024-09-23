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
import copy
from ..losses import accuracy
from ..losses.my_dice_loss import tversky
from ..losses.afma_loss import AFMALoss


@MODELS.register_module()
class AFMAHead(BaseDecodeHead):
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
                 patch_size=16,
                 **kwargs):
        self.patch_size = patch_size
        super().__init__(**kwargs)


        self.img_emb1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.img_conv1 = nn.Conv2d(64, self.num_classes, kernel_size=3, padding=1)

        self.partition = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

        self.f_conv1 = nn.Conv2d(64, self.num_classes, kernel_size=3, padding=1)

        self.resolution_trans = nn.Sequential(
            nn.Linear(patch_size * patch_size, 2 * patch_size * patch_size, bias=False),
            nn.Linear(2 * patch_size * patch_size, patch_size * patch_size, bias=False),
            nn.ReLU()
        )

        self.mask_conv1 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.mask_part = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

        self.loss_afma = AFMALoss(att_depth=0, out_channels=self.num_classes, patch_size=16)

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        self.activation = nn.Softmax(dim=1)
        self.att_depth = 0

    def forward(self, inputs, feats, img):
        """Forward function."""

        image_emb = self.img_emb1(img)
        image_conv1 = self.img_conv1(image_emb)

        feat = self.f_conv1(feats)

        attentions = []
        for i in range(image_conv1.shape[1]):
            unfold_img = self.partition(image_conv1[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_img = self.resolution_trans(unfold_img)

            unfold_feamap = self.partition(feat[:, i:i + 1, :, :])
            unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

            att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)
            att = torch.unsqueeze(att, 1)

            attentions.append(att)
        attentions = torch.cat((attentions), dim=1)

        x = inputs
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels,
                                     kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                                     stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels,
                                     bias=False)
        conv_feamap_size.weight = nn.Parameter(
            torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(x.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        correction = []

        x_argmax = torch.argmax(x, dim=1)

        pr_temp = torch.zeros(x.size()).to(x.device)
        src = torch.ones(x.size()).to(x.device)
        x_softmax = pr_temp.scatter(dim=1, index=x_argmax.unsqueeze(1), src=src)

        argx_feamap = conv_feamap_size(x_softmax) / (2 ** self.att_depth * 2 ** self.att_depth)
        fold_layer = torch.nn.Fold(output_size=(x.size()[-2], x.size()[-1]),
                                   kernel_size=(self.patch_size, self.patch_size),
                                   stride=(self.patch_size, self.patch_size))

        for i in range(x.size()[1]):
            non_zeros = torch.unsqueeze(torch.count_nonzero(attentions[:, i:i + 1, :, :], dim=-1) + 0.00001, dim=-1)

            att = torch.matmul(attentions[:, i:i + 1, :, :] / non_zeros,
                               torch.unsqueeze(self.unfold(argx_feamap[:, i:i + 1, :, :]), dim=1).transpose(-1, -2))

            att = torch.squeeze(att, dim=1)

            att = fold_layer(att.transpose(-1, -2))

            correction.append(att)

        correction = torch.cat(correction, dim=1)

        x = correction * x + x

        # x = self.activation(x)

        outs = x
        ###
        #   my custom
        ###
        '''
        m_conv1 = self.mask_conv1(inputs)
        m_part = []
        for i in range(m_conv1.shape[1]):
            unfold_m = self.mask_part(m_conv1[:, i:i + 1, :, :]).transpose(-1, -2)
            unfold_m = self.resolution_trans(unfold_m)
            unfold_m = unfold_m.unsqueeze(1)
            m_part.append(unfold_m)
        m_part = torch.cat((m_part), dim=1)

        att = torch.matmul(attentions, m_part) / (self.patch_size * self.patch_size)

        fold_layer = nn.Fold(output_size=(224, 224), kernel_size=self.patch_size, stride=self.patch_size)

        att = att.transpose(-1, -2)
        
        outs = []
        for i in range(att.shape[1]):
            a = att[:, i, :, :]
            fold_m = fold_layer(a)
            outs.append(fold_m)
        outs = torch.cat((outs), dim=1)

        outs = outs * inputs + outs

        outs = self.activation(outs)
        '''
        return outs, attentions

    def loss(self, inputs, batch_data_samples, feats, img) -> dict:
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

        seg_logits, attn = self.forward(inputs, feats, img)
        losses = self.loss_by_feat(seg_logits, batch_data_samples, attn)

        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType, feat, img) -> Tensor:
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
        seg_logits, attn = self.forward(inputs, feat, img)

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
                     batch_data_samples: SampleList,
                     attentions) -> dict:
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
        # dice = tversky(
        #         seg_logits, seg_label, valid_mask=(seg_label != self.ignore_index).long(), alpha=0.5, beta=0.5)
        # for i in range(seg_logits.shape[0]):
        #     total_dice = total_dice +dice[i]
        # # loss['seg_dice'] = total_dice/(seg_logits.shape[0] - 1) * 100
        # loss['seg_dice'] = total_dice
        total_dice = 0
        dice = tversky(
            seg_logits, seg_label, valid_mask=(seg_label != self.ignore_index).long(), alpha=0.5, beta=0.5)
        for i in range(1, seg_logits.shape[0]):
            total_dice = total_dice + dice[i]
        loss['seg_dice'] = total_dice / (seg_logits.shape[0] - 1) * 100

        loss_afma = self.loss_afma(seg_logits, seg_label, attentions)
        loss['loss_afma'] = loss_afma


        return loss
