# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/zejiangh/MILAN
from collections import OrderedDict
from typing import Optional, Tuple, Union, Dict
import copy
import numpy as np
import torch
import torch.nn.functional as F

from mmengine.logging import MMLogger
from torch import nn
from mmengine.model import BaseModel

from mmseg.registry import MODELS
from mmengine.registry import Registry
from mmseg.structures import SegDataSample
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

class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


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



class ViTforCLIP(nn.Module):
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
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 finetune=False,
                 average_targets: int = 1) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        #(Hight*Width)Chancel -> Patch(Embedding)
        self.embconv = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)

        scale = width ** -0.5
        self.class_embedding1 = nn.Parameter(scale * torch.randn(width*2))
        self.class_embedding2 = nn.Parameter(scale * torch.randn(width * 4))
        self.class_embedding3 = nn.Parameter(scale * torch.randn(width * 8))

        self.ln_pre1 = LayerNorm(width*2)
        self.ln_pre2 = LayerNorm(width*4)
        self.ln_pre3 = LayerNorm(width*8)

        self.finetune = finetune
        if finetune is False:
            self.ln_post1 = LayerNorm(width*2)
            self.ln_post2 = LayerNorm(width*4)
            self.ln_post3 = LayerNorm(width*8)
            self.proj1 = nn.Parameter(scale * torch.randn(width*2, width*2))
            self.proj2 = nn.Parameter(scale * torch.randn(width*4, width*4))
            self.proj3 = nn.Parameter(scale * torch.randn(width*8, width*8))

        self.average_targets = average_targets

        self.transformer = nn.ModuleList()
        self.transformer.append(TransformerForCLIP(width*2, layers, heads))
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=width, out_channels=width * 2, kernel_size=1)
        self.transformer.append(TransformerForCLIP(width*4, layers, heads))
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=width*2, out_channels=width * 4, kernel_size=1)
        self.transformer.append(TransformerForCLIP(width*8, layers, heads))
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=width*4, out_channels=width * 8, kernel_size=1)

        self.pos_embedding1 = nn.Parameter(scale * torch.randn(int(56 ** 2 + 1), width*2))
        self.pos_embedding2 = nn.Parameter(scale * torch.randn(int(28 ** 2 + 1), width*4))
        self.pos_embedding3 = nn.Parameter(scale * torch.randn(int(14 ** 2 + 1), width*8))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function."""
        skip_feat = []
        x = self.embconv(x)  # shape = [*, width, grid, grid]
        skip_feat.append(x)
        x = self.down1(x)   # n*96*56*56
        x = self.conv1(x)   # n*192*56*56

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
        skip_feat.append(x)

        x = self.down2(x)   # n*192*28*28
        x = self.conv2(x)   # n*384*28*28
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
        skip_feat.append(x)

        x = self.down3(x)   # n*384*28*28
        x = self.conv3(x)   # n*768*14*14
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding3.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embedding3.to(x.dtype)
        x = self.ln_pre3(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attention, z = self.transformer[2](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if self.proj3 is not None:
            x = x @ self.proj3
        self.ln_post3(x)
        x = x[:, :-1, :]

        return x, skip_feat


@MODELS.register_module()
#class CLIP(nn.Module):
class UCLIP(BaseModel):
    """CLIP.

    Args:
        embed_dim (int): The embedding dimension.
        image_resolution (int): The image size.
        vision_layers (int): The number of layers in the vision transformer.
        vision_width (int): The feature dimension in the vision transformer.
        vision_patch_size (int): The patch size in the vision transformer.
        context_length (int): The context length.
        vocab_size (int): The vocabulary size.
        transformer_width (int): The feature dimension in the text transformer.
        transformer_heads (int): The number of attention heads in the
            text transformer.
        transformer_layers (int): The number of layers in the text transformer.
        fineturn (bool): Whether to fineturn the model.
        average_target (bool): Whether to average the target.
    """

    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        finetune: bool = False,
        average_targets: int = 1,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,

    ) -> None:
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64   # normal vision_head=12
        self.visual = ViTforCLIP(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            finetune=finetune,
            average_targets=average_targets,
        )

        self.transformer = TransformerForCLIP(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # add
        self.txt_linear = nn.Linear(26, 196)
        self.text2_projection = nn.Parameter(
            torch.empty(transformer_width, 196))
        self.txt_linear2 = nn.Linear(26, 196)
        # addend
        self.initialize_parameters()
        self.init_cfg = copy.deepcopy(init_cfg)
        # self.init_weights()

        data_preprocessor = MODELS.build(data_preprocessor)
        self.data_preprocessor = data_preprocessor




    def initialize_parameters(self) -> None:
        """Initialize the parameters.

        The pretrained weight will override the initialized parameters by this
        function.
        """
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers)**-0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(
                self.text_projection, std=self.transformer.width**-0.5)

        if self.text2_projection is not None:
            nn.init.normal_(
                self.text2_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self) -> torch.Tensor:
        """Build the attention mask."""
        # lazily create causal attention mask, with full attention between the
        # vision tokens pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype."""
        return self.visual.conv1.weight.dtype

    def encode_image(self,
                     image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the image.

        Get the feature and attention mask from the last layer of the visual
        branch of CLIP.

        Args:
            image (torch.Tensor): The image tensor with shape NCHW.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The feature and attention mask.
        """
        x, attention = self.visual(image.type(self.dtype))
        # x = self.visual(image.type(self.dtype))
        return x, attention

    '''
    def encode_text(self, data_samples):
        stack_tensor = []
        for i in range(len(data_samples)):
            # tensor2d = data_samples[i].token
            # tensor1d = tensor2d.view(-1)
            stack_tensor.append(data_samples[i].token[0])

        device = torch.device('cuda:0')
        text = torch.stack(stack_tensor, dim=0)
        text = text.to(device)

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attention, z = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 取每一个示例（文本，n），按照句子的尾标识符的位置，取embedding映射表的特征值 [n,l,d] -> [n,d]
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    '''

    # modify
    def encode_text(self, data_samples):
        device = torch.device('cuda:0')
        result = []
        result_sm = []

        if data_samples == None:
            result = torch.zeros((1, 196, 768),device=device)
            result_sm = torch.zeros((1, 3, 768),device=device)
            return result, result_sm


        stack_list = []
        for m in range(len(data_samples)):
            stack_token = []
            for n in range(len(data_samples[m].token)):
                stack_token.append(data_samples[m].token[n])
            stack_list.append(stack_token)

        for i in range(len(stack_list)):
            text = torch.stack(stack_list[i], dim=0)
            text = text.to(device)

            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x, attention, z = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]

            x2 = x.clone()

            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # 取每一个示例（文本，n），按照句子的尾标识符的位置，取embedding映射表的特征值 [n,l,d] -> [n,d]
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

            x = x.permute(1, 0)
            x = self.txt_linear(x)
            x = x.permute(1, 0)

            result.append(x)

            x2 = x2[torch.arange(x2.shape[0]), text.argmax(dim=-1)] @ self.text2_projection
            result_sm.append(x2)

        result = torch.stack(result, dim=0)     # n*196*768
        result_sm = torch.stack(result_sm, dim=0)
        result_sm = result_sm[:, :3, :]

        return result, result_sm

    # modify
    def encode_text_for_simi(self, data_samples):
        stack_list = []
        for m in range(len(data_samples)):
            stack_token = []
            for n in range(len(data_samples[m].token)):
                stack_token.append(data_samples[m].token[n])
            stack_list.append(stack_token)

        device = torch.device('cuda:0')


        result = []
        for i in range(len(stack_list)):
            text = torch.stack(stack_list[i], dim=0)
            text = text.to(device)

            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x, attention, z = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]

            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # 取每一个示例（文本，n），按照句子的尾标识符的位置，取embedding映射表的特征值 [n,l,d] -> [n,d]
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text2_projection

            x = x.permute(1, 0)
            x = self.txt_linear(x)
            x = x.permute(1, 0)

            result.append(x)

        result = torch.stack(result, dim=0)     # n*26*196
        result = result[:, :3, :]

        return result


    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        if mode == 'tensor':
            image_features, text_features,_ ,_ = self.extract_feat(inputs, data_samples)
            return image_features

        elif mode == 'predict':
            feats = self.predict(inputs, data_samples)
            predictions = torch.argmax(feats, 1)
            for data_sample in data_samples:
                data_sample.pred_sem_seg = data_sample.gt_sem_seg
            predictions = data_samples
            return predictions

        elif mode == 'loss':
            loss = self.loss(inputs, data_samples)
            return dict(loss=loss)

    '''
    def calc_similarity(self, inputs, data_samples):
        x = self.encode_image(inputs)
        image_features = x[:, 0, :]
        text_features = self.encode_text(data_samples)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        # x = []
        # for i in range(len(image_features)):
        #     logits_per_image = logit_scale * image_features[i] @ text_features[i].t()
        #     logits_per_text = logits_per_image.t()
        #     x.append(logits_per_image)
        #     x.append(logits_per_text)
        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    '''

    # modify
    def calc_similarity(self, inputs, text_feat):
        x = inputs.clone()
        img_features = x.permute(0, 2, 1)
        img_features = img_features[:, :3, :]
        # img_features = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
        txt_features = text_feat

        logit_scale = self.logit_scale.exp()

        logits_per_image = []
        logits_per_text = []
        for i in range(x.shape[0]):
            # normalized features
            # 小心梯度传递回传时，
            image_features = img_features[i].clone()
            text_features = txt_features[i].clone()

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_per_image = logit_scale * image_features @ text_features.t()
            logit_per_text = logit_per_image.t()

            logits_per_image.append(logit_per_image)
            logits_per_text.append(logit_per_text)

        logits_per_image = torch.stack(logits_per_image, dim=0)
        logits_per_text = torch.stack(logits_per_text, dim=0)

        return logits_per_image, logits_per_text

    def extract_feat(self, inputs, data_samples):
        image_features, attn = self.encode_image(inputs)
        text_features, t2 = self.encode_text(data_samples)
        return image_features, text_features, t2, attn

    # modify
    def loss(self, x, t):
        logits_per_image, logits_per_text = self.calc_similarity(x, t)
        loss_fnc_img = nn.CrossEntropyLoss()
        loss_fnc_txt = nn.CrossEntropyLoss()
        device = torch.device('cuda:0')
        ground_truth = torch.arange(logits_per_image.shape[1], dtype=torch.long, device=device)

        losses = 0
        for i in range(logits_per_image.shape[0]):
            loss_img = loss_fnc_img(logits_per_image[i], ground_truth)
            loss_txt = loss_fnc_txt(logits_per_text[i], ground_truth)
            losses = losses + (loss_img + loss_txt) / 2

        total_loss = losses / len(logits_per_image)
        return total_loss

    # modify
    def predict(self, inputs, data_samples):
        x = inputs.clone()
        img_features = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        txt_features = self.encode_text_for_simi(data_samples)
        logit_scale = self.logit_scale.exp()

        for i in range(x.shape[0]):
            image_features = img_features[i]
            text_features = txt_features[i]

            image_features /= image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features * logit_scale

            pred_score = F.softmax(logits_per_image, dim=1)
            pred_label = pred_score.argmax(dim=1, keepdim=True).detach()

            data_samples[i].pred_scores = pred_score
            data_samples[i].pred_labels = pred_label

        return data_samples



    def train_step(self, data, optim_wrapper):

        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data):
        data = self.data_preprocessor(data)
        outputs = self._run_forward(data, mode='predict')
        return outputs

    def test_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        outputs = self(*data, mode='predict')
        return outputs



def build_clip_model(state_dict: dict,
                     finetune: bool = False,
                     average_targets: int = 1) -> nn.Module:
    """Build the CLIP model.

    Args:
        state_dict (dict): The pretrained state dict.
        finetune (bool): Whether to fineturn the model.
        average_targets (bool): Whether to average the target.

    Returns:
        nn.Module: The CLIP model.
    """
    vit = 'visual.proj' in state_dict

    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')
        ])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round(
            (state_dict['visual.positional_embedding'].shape[0] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split('.')[2] for k in state_dict
            if k.startswith('transformer.resblocks')))

    model = UCLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        finetune,
        average_targets,
    )

    for key in ['input_resolution', 'context_length', 'vocab_size']:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    MMLogger.get_current_instance().info(f'Load CLIP model: {msg}')
    if finetune == True:
        return model.train()
    else:
        return model.eval()
#MODELS = Registry('Clip', build_func=build_clip_model) # 在实例化注册器的时候，就要传入自定义的build方法


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

