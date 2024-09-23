import torch
import torch.nn as nn
from mmseg.registry import MODELS
from ..backbones.cliptest import TransformerForCLIP,LayerNorm
from typing import Optional, Tuple, Union, Dict
import numpy as np
import copy
from mmengine.runner.checkpoint import _load_checkpoint
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from safetensors.torch import load_file
# path = 'ckpt/epoch_200.pth'



@MODELS.register_module()
class ClipforSwin(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ) -> None:
        super().__init__()

        self.context_length = context_length

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

        # self.text_projection = nn.Parameter(
        #     torch.empty(transformer_width, embed_dim))
        self.text_projection = nn.Parameter(
            torch.empty(transformer_width, transformer_width))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # add

        # when class_num = 4
        # self.txt_linear = nn.Linear(18, 3)
        # self.txt_linear2 = nn.Linear(18, 49)

        # when class_num = 5
        # self.txt_linear = nn.Linear(24, 4)

        #self.img_linear = nn.Linear(50176, 768)
        # self.img_linear = nn.Linear(49, 3)
        #self.img_linear = nn.Conv2d(in_channels=3136,out_channels=768,kernel_size=1,stride=1)
        # addend
        # path = r'D:\PyCharmProject\LYM\Glip\mmsegmentation\ckpt\clip_txt.pth'
        path2 = r'D:\PyCharmProject\LYM\Glip\mmsegmentation\ckpt\open_clip_model.safetensors'
        # if path is not None:
        #     self.checkpoint = _load_checkpoint(path)
        self.checkpoint = load_file(path2)
        self.initialize_parameters()
        self._freeze()

    def initialize_parameters(self) -> None:
        """Initialize the parameters.

        The pretrained weight will override the initialized parameters by this
        function.
        """
        if self.checkpoint is not None:
            # state_dict = self.checkpoint['state_dict']
            state_dict = self.checkpoint
            load_state_dict(self, state_dict, strict=False, logger=None)
        else:
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
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)



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
        return self.conv1.weight.dtype

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

    # modify
    def encode_text(self, data_samples):
        device = torch.device('cuda:0')
        result = []

        if data_samples == None:
            result = torch.zeros((1, 196, 768),device=device)
            return result

        stack_list = []
        for batch in range(len(data_samples)):
            stack_token = []
            for i in range(len(data_samples[batch].token)):
                stack_token.append(data_samples[batch].token[i])
            stack_list.append(stack_token)

        for n in range(len(stack_list)):
            text = torch.stack(stack_list[n], dim=0)
            text = text.to(device)

            x = self.token_embedding(text) # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x, attention, z = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # 取每一个示例（文本，n），按照句子的尾标识符的位置，取embedding映射表的特征值 [n,l,d] -> [n,d]
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

            # x = x.permute(1, 0)
            # x = self.txt_linear(x)
            # x = x.permute(1, 0)

            result.append(x)

        result = torch.stack(result, dim=0)     # n*196*768
        return result

    # modify
    def calc_similarity(self, inputs, text_feat):
        x = inputs.clone()

        img_features = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
        img_features = self.img_linear(img_features)
        img_features = img_features.permute(0, 2, 1)

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
        # image_features = self.encode_image(inputs)
        text_features = self.encode_text(data_samples)
        return inputs, text_features

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                ) -> Union[Dict[str, torch.Tensor], list]:
        x, t = self.extract_feat(inputs, data_samples)
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

    def loss(self,inputs: torch.Tensor,
                data_samples: Optional[list] = None):
        total_loss = self.forward(inputs, data_samples)
        return dict(loss=total_loss)


    def encode_text2(self, data_samples):
        device = torch.device('cuda:0')
        result = []

        if data_samples == None:
            result = torch.zeros((1, 196, 768),device=device)
            return result

        stack_list = []
        for batch in range(len(data_samples)):
            stack_token = []
            for i in range(len(data_samples[batch].token)):
                stack_token.append(data_samples[batch].token[i])
            stack_list.append(stack_token)

        for n in range(len(stack_list)):
            text = torch.stack(stack_list[n], dim=0)
            text = text.to(device)

            x = self.token_embedding(text) # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x, attention, z = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # 取每一个示例（文本，n），按照句子的尾标识符的位置，取embedding映射表的特征值 [n,l,d] -> [n,d]
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

            x = x.permute(1, 0)
            x = self.txt_linear2(x)
            x = x.permute(1, 0)

            result.append(x)

        result = torch.stack(result, dim=0)     # n*196*768
        return result

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False