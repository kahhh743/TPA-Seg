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
from typing import Optional, Tuple, Union, Dict
from collections import OrderedDict
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
import torch.utils.checkpoint as cp


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



class QuickGELU(nn.Module):
    """A faster version of GELU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block (RAB).

    This module implements the same function as the MultiheadAttention,
    but with a different interface, which is mainly used
    in CLIP.

    Args:
        d_model (int): The feature dimension.
        n_head (int): The number of attention heads.
        attn_mask (torch.Tensor, optional): The attention mask.
            Defaults to None.
    """

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: Optional[torch.Tensor] = None,
                 return_attention: bool = False) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.return_attention = return_attention

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Attention function."""
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        if self.return_attention:
            return self.attn(
                x,
                x,
                x,
                need_weights=self.return_attention,
                attn_mask=self.attn_mask)
        else:
            return self.attn(
                x,
                x,
                x,
                need_weights=self.return_attention,
                attn_mask=self.attn_mask)[0]

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward function."""
        if self.return_attention:
            x_, attention = self.attention(self.ln_1(x))
            x = x + x_
            x = x + self.mlp(self.ln_2(x))
            return x, attention
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class TransformerForCLIP(nn.Module):
    """TransformerForCLIP.

    Both visual and text branches use this transformer.

    Args:
        width (int): The feature dimension.
        layers (int): The number of layers.
        heads (int): The number of attention heads.
        attn_mask (torch.Tensor, optional): The attention mask.
    """

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList()

        for i in range(layers - 1):
            self.resblocks.append(
                ResidualAttentionBlock(width, heads, attn_mask))
        self.resblocks.append(
            ResidualAttentionBlock(
                width, heads, attn_mask, return_attention=True))


    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward function."""

        z = []
        for idx, blk in enumerate(self.resblocks):
            if idx < self.layers - 1:
                x = blk(x)
                z.append(x.permute(1, 0, 2))
            else:
                x, attention = blk(x)
                z.append(x.permute(1, 0, 2))

        return x, attention, z

class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 *,
                 kernel_size=4,
                 scale_factor=2):
        super().__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out

class DecodeViTforCLIP(nn.Module):
    """Vision Transformer for CLIP.

    Args:
        input_resolution (int): The image size.
        patch_size (int): The patch size.
        width (int): The feature dimension.
        layers (int): The number of layers.
        heads (int): The number of attention heads.
        out_dim (int): The output dimension.
        fineturn (bool): Whether to fineturn the model.
        average_target (bool): Whether to average the target.
    """

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 finetune=False,
                 average_targets: int = 1) -> None:
        super().__init__()

        scale = width ** -0.5
        self.class_embedding1 = nn.Parameter(scale * torch.randn(width*4))
        self.class_embedding2 = nn.Parameter(scale * torch.randn(width * 2))
        self.class_embedding3 = nn.Parameter(scale * torch.randn(width))

        self.ln_pre1 = LayerNorm(width*4)
        self.ln_pre2 = LayerNorm(width*2)
        self.ln_pre3 = LayerNorm(width)

        self.finetune = finetune
        if finetune is False:
            self.ln_post1 = LayerNorm(width*4)
            self.ln_post2 = LayerNorm(width*2)
            self.ln_post3 = LayerNorm(width)
            self.proj1 = nn.Parameter(scale * torch.randn(width*4, width*4))
            self.proj2 = nn.Parameter(scale * torch.randn(width*2, width*2))
            self.proj3 = nn.Parameter(scale * torch.randn(width, width))

        self.average_targets = average_targets

        self.transformer = nn.ModuleList()
        self.transformer.append(TransformerForCLIP(width*4, layers, heads))
        self.up1 = nn.ConvTranspose2d(width*8, width*8, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=width*8, out_channels=width * 4, kernel_size=1)
        self.transformer.append(TransformerForCLIP(width*2, layers, heads))
        self.up2 = nn.ConvTranspose2d(width*4, width*4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=width*4, out_channels=width * 2, kernel_size=1)
        self.transformer.append(TransformerForCLIP(width, layers, heads))
        self.up3 = nn.ConvTranspose2d(width*2, width*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=width*2, out_channels=width, kernel_size=1)

        self.pos_embedding1 = nn.Parameter(scale * torch.randn(int(28 ** 2 + 1), width*4))
        self.pos_embedding2 = nn.Parameter(scale * torch.randn(int(56 ** 2 + 1), width*2))
        self.pos_embedding3 = nn.Parameter(scale * torch.randn(int(112 ** 2 + 1), width))

        num_class = 3
        self.up4 = nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=width, out_channels=width//2, kernel_size=1)
        self.linear = nn.Linear(in_features=96, out_features=3)

        self.linear1 = nn.Conv2d(in_channels=width*8, out_channels=width*4, kernel_size=1)
        self.linear2 = nn.Conv2d(in_channels=width*4, out_channels=width*2, kernel_size=1)
        self.linear3 = nn.Conv2d(in_channels=width*2, out_channels=width, kernel_size=1)

    def forward(self, x: torch.Tensor, attn) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function."""
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], int(x.shape[2] ** 0.5), int(x.shape[2] ** 0.5))

        x = self.up1(x)   # n*768*28*28
        x = self.conv1(x)   # n*384*28*28
        x = torch.cat((x, attn[2]), dim=1)  # n*768*28*28
        x = self.linear1(x)

        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding1.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embedding1.to(x.dtype)
        x = self.ln_pre1(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attention, z = self.transformer[0](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if self.proj1 is not None:
            x = x @ self.proj1
        self.ln_post1(x)
        x = x[:, :-1, :]
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], int(x.shape[2] ** 0.5), int(x.shape[2] ** 0.5))

        x = self.up2(x)   # n*384*56*56
        x = self.conv2(x)   # n*192*56*56
        x = torch.cat((x,attn[1]),dim=1)  # n*384*56*56
        x = self.linear2(x)

        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding2.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embedding2.to(x.dtype)
        x = self.ln_pre2(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attention, z = self.transformer[1](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if self.proj2 is not None:
            x = x @ self.proj2
        self.ln_post2(x)
        x = x[:, :-1, :]
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], int(x.shape[2] ** 0.5), int(x.shape[2] ** 0.5))

        x = self.up3(x)   # n*192*112*112
        x = self.conv3(x)   # n*96*112*112
        x = torch.cat((x,attn[0]),dim=1)  # n*384*112*112
        x = self.linear3(x) # n*96*112*112

        # x = x.reshape(x.shape[0], x.shape[1],
        #               -1)  # shape = [*, width, grid ** 2]
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat([
        #     self.class_embedding3.to(x.dtype) + torch.zeros(
        #         x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        # ],
        #     dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.pos_embedding3.to(x.dtype)
        # x = self.ln_pre3(x)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x, attention, z = self.transformer[2](x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        # if self.proj3 is not None:
        #     x = x @ self.proj3
        # self.ln_post3(x)
        # x = x[:, :-1, :]

        x = self.up4(x)   # n*96*224*224
        # x = self.conv4(x)   # n*48*224*224

        # x = x.permute(0, 2, 3, 1)
        # x = self.linear(x)

        return x

@MODELS.register_module()
class MyTransUHead(BaseDecodeHead):
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
                 width=96,
                 layers=1,
                 heads=12,
                 **kwargs):
        self.width = width
        self.layers = layers
        self.heads = heads
        self.channels = 48
        self.out_channels = 3

        super().__init__(**kwargs)

        self.decode = DecodeViTforCLIP(width=96, layers=1, heads=12)

    def forward(self, inputs, attn):
        """Forward function."""
        output = self.decode(inputs, attn)
        output = self.cls_seg(output)
        return output

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList, attn) -> dict:
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
        seg_logits = self.forward(inputs, attn)

        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict], attn) -> Tensor:
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
        seg_logits = self.forward(inputs, attn)

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