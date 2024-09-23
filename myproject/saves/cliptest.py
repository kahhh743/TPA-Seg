# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/zejiangh/MILAN
from collections import OrderedDict
from typing import Optional, Tuple, Union, Dict
import copy
import numpy as np
import torch
import torch.nn.functional as F
from ..utils import resize

from mmengine.logging import MMLogger
from torch import nn
from mmengine.model import BaseModel

from mmseg.registry import MODELS
from mmengine.registry import Registry
from ..losses.cross_entropy_loss import cross_entropy

@MODELS.register_module()
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


@MODELS.register_module()
class QuickGELU(nn.Module):
    """A faster version of GELU."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return x * torch.sigmoid(1.702 * x)

@MODELS.register_module()
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

@MODELS.register_module()
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
        for _ in range(layers - 1):
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


@MODELS.register_module()
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
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = TransformerForCLIP(width, layers, heads)

        self.finetune = finetune
        if finetune is False:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.average_targets = average_targets

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function."""
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attention, z = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        #x = self.ln_post(x)
        # 取x的L这个维度的第一个向量，即class token  x.shape=[batch_size, width]
        # x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        self.ln_post(x)

        return x, attention

@MODELS.register_module()
#class CLIP(nn.Module):
class CLIP(BaseModel):
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
        self.text_projection2 = nn.Parameter(
            torch.empty(transformer_width, 196))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        self.init_cfg = copy.deepcopy(init_cfg)
        # self.init_weights()

        data_preprocessor = MODELS.build(data_preprocessor)
        self.data_preprocessor = data_preprocessor

        self.txt_linear = nn.Linear(26, 196)

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
        return x

    def encode_label(self, data_samples):
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

    def encode_text_neck(self, data_samples):
        stack_list = []
        for m in range(len(data_samples)):
            stack_token = []
            for n in range(len(data_samples[m].token)):
                stack_token.append(data_samples[m].token[n])
            stack_list.append(stack_token)

        device = torch.device('cuda:0')


        result_neck = []
        # result_simi = []
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
            # y = x.clone()
            # y = y[torch.arange(y.shape[0]), text.argmax(dim=-1)] @ self.text_projection2

            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection


            x = x.permute(1, 0)     # 768*26
            x = self.txt_linear(x)      # 768*196
            x = x.permute(1, 0)     # 196*768

            result_neck.append(x)
            # result_simi.append(y)
            
        result_neck = torch.stack(result_neck, dim=0)     # n*196*768
        # result_simi = torch.stack(result_simi, dim=0)       # n*26*196

        return result_neck

    def encode_text_simi(self, data_samples):
        stack_list = []
        for m in range(len(data_samples)):
            stack_token = []
            for n in range(len(data_samples[m].token)):
                stack_token.append(data_samples[m].token[n])
            stack_list.append(stack_token)

        device = torch.device('cuda:0')

        # result_neck = []
        result_simi = []
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
            # y = x.clone()
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection2

            # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

            x = x.permute(1, 0)  # 768*26
            x = self.txt_linear(x)  # 768*196
            x = x.permute(1, 0)  # 196*768

            # result_neck.append(x)
            result_simi.append(x)

        # result_neck = torch.stack(result_neck, dim=0)  # n*196*768
        result_simi = torch.stack(result_simi, dim=0)       # n*26*196

        return result_simi

    def encode_text(self, data_samples):
        stack_list = []
        for m in range(len(data_samples)):
            stack_token = []
            for n in range(len(data_samples[m].token)):
                stack_token.append(data_samples[m].token[n])
            stack_list.append(stack_token)

        device = torch.device('cuda:0')

        result_neck = []
        result_simi = []
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
            # y = x.clone()
            # y = y[torch.arange(y.shape[0]), text.argmax(dim=-1)] @ self.text_projection2

            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

            x = x.permute(1, 0)  # 768*26
            x = self.txt_linear(x)  # 768*196
            x = x.permute(1, 0)  # 196*768

            result_neck.append(x)
            # result_simi.append(y)

        result_neck = torch.stack(result_neck, dim=0)  # n*196*768
        # result_simi = torch.stack(result_simi, dim=0)       # n*26*196

        return result_neck


    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        if mode == 'tensor':
            image_features, text_features = self.extract_feat(inputs, data_samples)
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

    def calc_similarity(self, inputs, data_samples, text_feat):
        x = self.encode_image(inputs)
        x = x.view(inputs.shape[0], inputs.shape[1], inputs.shape[2]*inputs.shape[3])  # n*2*196

        text_feat = text_feat[:, :x.shape[1], :]   # n*26*196

        logits_per_image = []
        logits_per_text = []


        for i in range(len(x)):
            # normalized features
            image_features = x[i]
            text_features = text_feat[i]
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logit_per_image = logit_scale * image_features @ text_features.t()
            logit_per_text = logit_per_image.t()
            logits_per_image.append(logit_per_image)
            logits_per_text.append(logit_per_text)

        return logits_per_image, logits_per_text

    def extract_feat(self, inputs, data_samples):
        image_features = self.encode_image(inputs)
        text_features = self.encode_label(data_samples)
        return image_features, text_features

    def loss(self, inputs, data_samples, text_feat):

        # gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in data_samples]
        # seg_label = torch.stack(gt_semantic_segs, dim=0)
        # seg_label = seg_label.view(seg_label.shape[0], seg_label.shape[2], seg_label.shape[3])
        # loss_mask = cross_entropy(inputs, seg_label, ignore_index=255)

        logits_per_image, logits_per_text = self.calc_similarity(inputs, data_samples, text_feat)



        loss_fnc_img = nn.CrossEntropyLoss()
        loss_fnc_txt = nn.CrossEntropyLoss()
        device = torch.device('cuda:0')
        ground_truth = torch.arange(len(inputs), dtype=torch.long, device=device)

        batch = len(logits_per_image)
        loss = 0
        for i in range(batch):
            loss_img = loss_fnc_img(logits_per_image[i], ground_truth)
            loss_txt = loss_fnc_txt(logits_per_text[i], ground_truth)
            loss = loss + (loss_img + loss_txt) / 2

        total_loss = loss / batch
        return total_loss

    def predict(self, seg_logits, data_samples):

        inputs = resize(
            seg_logits,
            size=[224, 224],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False).squeeze(0)

        image_features, _ = self.encode_image(inputs)
        text_features = self.encode_label(data_samples)

        # adding
        '''
        image_features = self.extract_image_feat(images=images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = image_features @ self.text_prototype_embeds.to(
            image_features.device) * self.logit_scale.exp()

        pred_scores = F.softmax(logits_per_image, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()
            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples
        '''
        return text_features

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

    model = CLIP(
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

