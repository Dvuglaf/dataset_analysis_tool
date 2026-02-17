from pathlib import Path
from functools import partial
from typing import Any
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torch.cuda.amp import autocast
from torchvision import transforms

from timm.models import register_model
from timm.models.layers import drop_path, to_2tuple
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
import sys

sys.path.append('..')

from metrics.base import BaseMetric, Metric


# TODO: re-add positional encoding helpers only if VideoMAE feature extractor is re-used elsewhere
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class BaseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self,
                     inputs: Tensor,
                     outputs: Tensor
                     ) -> Tensor:
        """Computes the loss between the inputs and outputs using CrossEntropyLoss.

        Args:
            inputs (Tensor): The ground truth labels.
            outputs (Tensor): The predicted outputs from the model.

        Returns:
            Tensor: The computed CrossEntropyLoss.
        """

        return self.loss_fn(inputs, outputs)

    def preprocess_batch(self,
                         batch: Tensor | dict[Any, Tensor] | tuple[Tensor, ...] | list[Tensor],
                         device: torch.device | str
                         ) -> Tensor | dict[Any, Tensor] | tuple[Tensor, ...] | list[Tensor]:
        """Preprocesses a batch of data by moving it to the specified device.

        Args:
            batch (Tensor | dict | tuple | list): The batch of data to preprocess.
                It can be a single Tensor, a dictionary of Tensors, or a list/tuple of Tensors.
            device (torch.device | str): The device to move the batch to.

        Returns:
            Tensor | dict | tuple | list: The batch moved to the specified device.

        Raises:
            TypeError: If the batch is not a Tensor, dictionary, list, or tuple.
        """
        if isinstance(batch, Tensor):
            return batch.to(device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        elif isinstance(batch, tuple):
            return tuple(x.to(device, non_blocking=True) for x in batch)
        elif isinstance(batch, list):
            return list(x.to(device, non_blocking=True) for x in batch)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def init_weights(self, path_to_checkpoint):
        pass  # TODO

    def save_weights(self,
                     path_to_checkpoint: str | Path,
                     epoch: int,
                     optimizer: torch.optim.Optimizer,
                     loss_scaler
                     ):
        pass  # TODO


class PretrainVideoMAEV2Configuration:
    model_type = 'pretrain_videomaev2'

    def __init__(self,
                 all_frames=16,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer: partial[nn.LayerNorm] = nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 with_cp=False,
                 cos_attn=False,
                 normalize_target=True,
                 mask_type: str = "tube",
                 mask_ratio: float = 0.9,
                 decoder_mask_type: str = "run_cell",
                 decoder_mask_ratio: float = 0.0,
                 ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder_in_chans = encoder_in_chans
        self.encoder_num_classes = encoder_num_classes

        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_classes = decoder_num_classes  # decoder_num_classes=768
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.init_values = init_values
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.tubelet_size = tubelet_size
        self.with_cp = with_cp
        self.all_frames = all_frames
        self.cos_attn = cos_attn
        self.normalize_target = normalize_target
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.decoder_mask_type = decoder_mask_type
        self.decoder_mask_ratio = decoder_mask_ratio


class FinetuneVideoMAEV2Configuration:
    model_type = 'finetune_videomaev2'

    def __init__(self,
                 num_classes,
                 all_frames=16,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 head_drop_rate=0.,
                 norm_layer: partial[nn.LayerNorm] = nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 with_cp=False,
                 cos_attn=False
                 ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.head_drop_rate = head_drop_rate
        self.norm_layer = norm_layer
        self.init_values = init_values
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.init_scale = init_scale
        self.all_frames = all_frames
        self.tubelet_size = tubelet_size
        self.use_mean_pooling = use_mean_pooling
        self.with_cp = with_cp
        self.cos_attn = cos_attn


CONFIGS = {
    "pretrain_videomaev2_small_patch16_224":
        PretrainVideoMAEV2Configuration(
            all_frames=16,
            img_size=224,
            patch_size=16,
            encoder_embed_dim=384,
            encoder_depth=12,
            encoder_num_heads=6,
            encoder_num_classes=0,
            decoder_num_classes=1536,  # 16 * 16 * 3 * 2
            decoder_embed_dim=192,
            decoder_num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
    "pretrain_videomaev2_base_patch16_224":
        PretrainVideoMAEV2Configuration(
            all_frames=16,
            img_size=224,
            patch_size=16,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            encoder_num_classes=0,
            decoder_num_classes=1536,  # 16 * 16 * 3 * 2
            decoder_embed_dim=384,
            decoder_num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ),
    "pretrain_videomaev2_large_patch16_224":
        PretrainVideoMAEV2Configuration(
            all_frames=16,
            img_size=224,
            patch_size=16,
            encoder_embed_dim=1024,
            encoder_depth=24,
            encoder_num_heads=16,
            encoder_num_classes=0,
            decoder_num_classes=1536,  # 16 * 16 * 3 * 2
            decoder_embed_dim=512,
            decoder_num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
    "pretrain_videomaev2_huge_patch16_224":
        PretrainVideoMAEV2Configuration(
            all_frames=16,
            img_size=224,
            patch_size=16,
            encoder_embed_dim=1280,
            encoder_depth=32,
            encoder_num_heads=16,
            encoder_num_classes=0,
            decoder_num_classes=1536,  # 16 * 16 * 3 * 2
            decoder_embed_dim=512,
            decoder_num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
    "pretrain_videomaev2_giant_patch14_224":
        PretrainVideoMAEV2Configuration(
            all_frames=16,
            img_size=224,
            patch_size=14,
            encoder_embed_dim=1408,
            encoder_depth=40,
            encoder_num_heads=16,
            encoder_num_classes=0,
            decoder_num_classes=1176,  # 14 * 14 * 3 * 2,
            decoder_embed_dim=512,
            decoder_num_heads=8,
            mlp_ratio=48 / 11,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
    "finetune_videomaev2_small_patch16_224":
        FinetuneVideoMAEV2Configuration(
            num_classes=400,
            all_frames=16,
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
    "finetune_videomaev2_base_patch16_224":
        FinetuneVideoMAEV2Configuration(
            num_classes=400,
            all_frames=16,
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
    "finetune_videomaev2_large_patch16_224":
        FinetuneVideoMAEV2Configuration(
            num_classes=400,
            all_frames=16,
            img_size=224,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
    "finetune_videomaev2_huge_patch16_224":
        FinetuneVideoMAEV2Configuration(
            num_classes=400,
            all_frames=16,
            img_size=224,
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
    "finetune_videomaev2_giant_patch14_224":
        FinetuneVideoMAEV2Configuration(
            num_classes=400,
            all_frames=16,
            img_size=224,
            patch_size=16,
            embed_dim=1408,
            depth=40,
            num_heads=16,
            mlp_ratio=48 / 11,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ),
}


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CosAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # self.scale = qk_scale or head_dim**-0.5
        # DO NOT RENAME [self.scale] (for no weight decay)
        if qk_scale is None:
            self.scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))),
                requires_grad=True)
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (
                F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        # torch.log(torch.tensor(1. / 0.01)) = 4.6052
        logit_scale = torch.clamp(self.scale, max=4.6052).exp()

        attn = attn * logit_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 cos_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if cos_attn:
            self.attn = CosAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 num_frames=16,
                 tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_spatial_patches = (img_size[0] // patch_size[0]) * (
                img_size[1] // patch_size[1])
        num_patches = num_spatial_patches * (num_frames // tubelet_size)

        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        # Need to permute (B, T, C, H, W) -> (B, C, T, H, W),
        # as conv3d expects temporal dimension to be at index 2
        x = x.permute(0, 2, 1, 3, 4)
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})."
        )

        # b, c, l -> b, l, c
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VideoMAEv2ForFinetune(BaseModel):
    def __init__(self, config: FinetuneVideoMAEV2Configuration):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            num_frames=config.all_frames,
            tubelet_size=config.tubelet_size
        )
        num_patches = self.patch_embed.num_patches

        if config.use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
        else:
            # sine-cosine.py positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, config.embed_dim)

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=config.norm_layer,
                init_values=config.init_values,
                cos_attn=config.cos_attn) for i in range(config.depth)
        ])
        self.norm = nn.Identity() if config.use_mean_pooling else config.norm_layer(config.embed_dim)
        self.fc_norm = config.norm_layer(config.embed_dim) if config.use_mean_pooling else None
        self.head_dropout = nn.Dropout(config.head_drop_rate)
        self.head = nn.Linear(config.embed_dim, config.num_classes) if config.num_classes > 0 else nn.Identity()

        if config.use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        self.head.weight.data.mul_(config.init_scale)
        self.head.bias.data.mul_(config.init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.size(0)

        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.config.with_cp:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return self.norm(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_dropout(x)
        x = self.head(x)
        return x


@register_model
def finetune_videomaev2_small_patch16_224():
    return VideoMAEv2ForFinetune(CONFIGS["finetune_videomaev2_small_patch16_224"])


@register_model
def finetune_videomaev2_base_patch16_224():
    return VideoMAEv2ForFinetune(CONFIGS["finetune_videomaev2_base_patch16_224"])


@register_model
def finetune_videomaev2_large_patch16_224():
    return VideoMAEv2ForFinetune(CONFIGS["finetune_videomaev2_large_patch16_224"])


@register_model
def finetune_videomaev2_huge_patch16_224():
    return VideoMAEv2ForFinetune(CONFIGS["finetune_videomaev2_huge_patch16_224"])


@register_model
def finetune_videomaev2_giant_patch14_224():
    return VideoMAEv2ForFinetune(CONFIGS["finetune_videomaev2_giant_patch14_224"])


class VideoMAEFeatureMetric(BaseMetric):
    """Извлекает признаки VideoMAE для фиксированных клипов и сохраняет их как массив."""

    def __init__(
            self,
            model_path: str,
            model_key: str = "finetune_videomaev2_giant_patch14_224",
            sequence_size: int = 16,
            batch_size: int = 8,
            resize_hw: tuple[int, int] | None = (224, 224),
            center_crop: bool = True,
            device_id: str = "cpu"
    ):
        self.model_path = Path(model_path)
        self.model_key = model_key
        self.sequence_size = max(1, sequence_size)
        self.batch_size = max(1, batch_size)
        self.resize_hw = resize_hw
        self.center_crop = center_crop
        self.device_id = device_id

        self.model: VideoMAEv2ForFinetune | None = None
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _compute_step(self, total_frames: int) -> int:
        """
        Вычисляет step на основе количества кадров в видео.

        Нужно минимум sequence_size кадров. Если кадров достаточно,
        выбираем step 2 или 3.

        Args:
            total_frames: общее количество кадров в видео

        Returns:
            step в диапазоне [1, 3]
        """
        if total_frames >= self.sequence_size * 3:
            return 3
        if total_frames >= self.sequence_size * 2:
            return 2
        return 1

    @property
    def name(self) -> str:
        return "videomae_features"

    def compute(self, video_path: Path) -> Metric:
        self._ensure_model()

        # Получаем количество кадров и вычисляем step
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        step = self._compute_step(total_frames)

        sequences_buffer: list[list[np.ndarray]] = []
        current_sequence: list[np.ndarray] = []
        feature_chunks: list[np.ndarray] = []
        read_any_frame = False
        stride = max(1, self.sequence_size // 2)

        def flush_sequences() -> None:
            nonlocal sequences_buffer
            if not sequences_buffer:
                return
            feature_chunks.append(self._extract_features(sequences_buffer))
            sequences_buffer = []

        for frame_idx, frame in enumerate(frame_generator(video_path)):
            read_any_frame = True
            if frame_idx % step != 0:
                continue

            processed = self._prepare_frame(frame)
            current_sequence.append(processed)

            if len(current_sequence) == self.sequence_size:
                sequences_buffer.append(current_sequence.copy())
                if stride >= self.sequence_size:
                    current_sequence = []
                else:
                    current_sequence = current_sequence[stride:]

            if len(sequences_buffer) >= self.batch_size:
                flush_sequences()

        flush_sequences()

        if not read_any_frame:
            raise ValueError(f"Не удалось прочитать кадры видео {video_path}")

        features = np.concatenate(feature_chunks, axis=0) if feature_chunks else self._empty_features()

        metric = Metric(name=self.name, value=features, is_array=True)
        return MetricsResult(name=self.name, metrics=[metric])

    # region helpers
    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self.model_key not in CONFIGS:
            raise ValueError(f"Неизвестная конфигурация модели: {self.model_key}")

        config = CONFIGS[self.model_key]
        model = VideoMAEv2ForFinetune(config)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Checkpoint не найден: {self.model_path}")

        state_dict = torch.load(self.model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device_id)
        model.eval()
        self.model = model

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        img = frame
        if self.center_crop:
            img = self._center_square_crop(img)
        if self.resize_hw is not None:
            w, h = self.resize_hw[1], self.resize_hw[0]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img

    @staticmethod
    def _center_square_crop(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h == w:
            return img
        side = min(h, w)
        y_start = (h - side) // 2
        x_start = (w - side) // 2
        return img[y_start:y_start + side, x_start:x_start + side]


    def _batch_sequences(self, sequences: list[list[np.ndarray]]) -> list[list[list[np.ndarray]]]:
        return [
            sequences[i:i + self.batch_size]
            for i in range(0, len(sequences), self.batch_size)
        ]

    def _preprocess_batch(self, batch: list[list[np.ndarray]]) -> torch.Tensor:
        array = np.asarray(batch, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array)
        tensor = tensor.permute(0, 1, 4, 2, 3)
        b, n, c, h, w = tensor.shape
        tensor = tensor.view(b * n, c, h, w)
        tensor = self.norm(tensor)
        tensor = tensor.view(b, n, c, h, w)
        return tensor

    def _extract_features(self, sequences: list[list[np.ndarray]]) -> np.ndarray:
        batches = self._batch_sequences(sequences)
        outputs: list[np.ndarray] = []
        autocast_ctx = autocast()
        for batch in batches:
            tensor = self._preprocess_batch(batch).to(self.device_id, non_blocking=True)
            with torch.no_grad():
                with autocast_ctx:
                    feats = self.model.forward_features(tensor)
            outputs.append(feats.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else self._empty_features()

    def _empty_features(self) -> np.ndarray:
        self._ensure_model()
        return np.zeros((0, self.model.config.embed_dim), dtype=np.float32)
    # endregion

