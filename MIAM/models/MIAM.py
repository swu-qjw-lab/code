import os
import sys
from operator import add
from functools import reduce, partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from timm.models.layers import DropPath, trunc_normal_
import torchvision.models as models

from models.feature_backbones import resnet
from models.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow

r'''
Modified timm library Vision Transformer implementation
https://github.com/rwightman/pytorch-image-models
'''


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, drop_path_rate=0., norm_layer=None):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

    def forward(self, im):
        x_clone = im.clone()
        B, _, H, W = im.shape
        patch_list = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                x = im[:, :, i * self.patch_size: i * self.patch_size + self.patch_size,
                    j * self.patch_size: j * self.patch_size + self.patch_size]
                patch_list.append(x)
        y = torch.stack(patch_list, dim=2)  # [2,8,64,32,32]
        y = y.transpose(1, 2)  # [2, 64, 8, 1024]
        y = y.flatten(3, 4)
        z = y

        return z


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiscaleBlock(nn.Module):

    def __init__(self, dim, patch_size, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.patch_size = patch_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # patch_32
        if self.patch_size[0] == 32:
            self.patch_emd = PatchEmbedding([256, 384], patch_size[0])
            self.norm5 = norm_layer(patch_size[0] * patch_size[0])
            self.pat_attn = Attention(patch_size[0] * patch_size[0], num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                      proj_drop=drop)
            self.norm6 = norm_layer(dim)
            self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.weitght_1 = nn.Parameter(torch.tensor(1/3, requires_grad=False))
        # patch_16
        if self.patch_size[1] == 16:
            self.patch_emd_16 = PatchEmbedding([256, 384], patch_size[1])
            self.norm7 = norm_layer(patch_size[1] * patch_size[1])
            self.pat_attn_16 = Attention(patch_size[1] * patch_size[1], num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                         proj_drop=drop)
            self.norm8 = norm_layer(dim)
            self.mlp4 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.weitght_2 = nn.Parameter(torch.tensor(1/3, requires_grad=False))
        # patch_8
        if self.patch_size[2] == 8:
            self.patch_emd_8 = PatchEmbedding([256, 384], patch_size[2])
            self.pat_attn_8 = Attention(patch_size[2] * patch_size[2], num_heads=8, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                        proj_drop=drop)
            self.norm9 = norm_layer(patch_size[2] * patch_size[2])
            self.norm10 = norm_layer(dim)
            self.mlp5 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.weitght_3 = nn.Parameter(torch.tensor(1/3, requires_grad=False))

        #CATs Inter-Correlation Self-Attention
        self.attn_multiscale = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.num_level = 8

    def forward(self, x):
        '''
        Multi-level aggregation
        '''
        B, N, H, W = x.shape

        if N == 1:
            x = x.flatten(0, 1)
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x.view(B, N, H, W)
        x = x.flatten(0, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.view(B, N, H, W)

        res = x.clone()
        patch_16 = x.clone()
        patch_8 = x.clone()

        sum = 0
        if self.patch_size[0] == 32:
            patch_dim = self.patch_size[0] * self.patch_size[0]
            patch_32 = self.patch_emd(x)
            patch_32 = patch_32.flatten(0, 1)
            patch_32 = self.drop_path(self.pat_attn(self.norm5(patch_32)))
            patch_32 = patch_32.view(B, 96, self.num_level, patch_dim)
            patch_32 = patch_32.transpose(1, 2)
            patch_32 = patch_32.view(B, self.num_level, 8, 12, self.patch_size[0], self.patch_size[0])
            patch_32 = patch_32.transpose(3, 4)
            patch_32 = patch_32.reshape(B, self.num_level, 256, 384) + res
            patch_32 = patch_32 + self.drop_path(self.mlp3(self.norm6(patch_32)))
            sum = torch.mul(self.weitght_1, patch_32) + sum
        if self.patch_size[1] == 16:
            patch_dim = self.patch_size[1] * self.patch_size[1]
            patch_16 = self.patch_emd_16(patch_16)
            patch_16 = patch_16.flatten(0, 1)
            patch_16 = self.drop_path(self.pat_attn_16(self.norm7(patch_16)))
            patch_16 = patch_16.view(B, 384, self.num_level, patch_dim)
            patch_16 = patch_16.transpose(1, 2)
            patch_16 = patch_16.view(B, self.num_level, 16, 24, self.patch_size[1], self.patch_size[1])
            patch_16 = patch_16.transpose(3, 4)
            patch_16 = patch_16.reshape(B, self.num_level, 256, 384) + res
            patch_16 = patch_16 + self.drop_path(self.mlp4(self.norm8(patch_16)))
            sum = torch.mul(self.weitght_2, patch_16) + sum
        if self.patch_size[2] == 8:
            patch_dim = self.patch_size[2] * self.patch_size[2]
            patch_8 = self.patch_emd_8(patch_8)
            patch_8 = patch_8.flatten(0, 1)
            patch_8 = self.drop_path(self.pat_attn_8(self.norm9(patch_8)))
            patch_8 = patch_8.view(B, 1536, self.num_level, patch_dim)
            patch_8 = patch_8.transpose(1, 2)
            patch_8 = patch_8.view(B, self.num_level, 32, 48, self.patch_size[2], self.patch_size[2])
            patch_8 = patch_8.transpose(3, 4)
            patch_8 = patch_8.reshape(B, self.num_level, 256, 384) + res
            patch_8 = patch_8 + self.drop_path(self.mlp5(self.norm10(patch_8)))
            sum = torch.mul(self.weitght_3,patch_8) + sum

        return sum


class TransformerAggregator(nn.Module):
    def __init__(self, num_hyperpixel, patch_size,img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed_x = nn.Parameter(torch.zeros(1, num_hyperpixel, 1, img_size, embed_dim // 2))
        self.pos_embed_y = nn.Parameter(torch.zeros(1, num_hyperpixel, img_size, 1, embed_dim // 2))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,patch_size=patch_size)
            for i in range(depth)])
        self.proj = nn.Linear(embed_dim, img_size ** 2)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed_x, std=.02)
        trunc_normal_(self.pos_embed_y, std=.02)
        self.apply(self._init_weights)

        self.num_level = 8
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_hyperpixel, num_hyperpixel // self.num_level, kernel_size=1),
            nn.BatchNorm2d(num_hyperpixel // self.num_level),
            nn.GELU(),
            nn.Conv2d(num_hyperpixel // self.num_level, num_hyperpixel, kernel_size=1),
        )

        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(self.num_level, num_hyperpixel // self.num_level, kernel_size=1),
            nn.BatchNorm2d(num_hyperpixel // self.num_level),
            nn.GELU(),
            nn.Conv2d(num_hyperpixel // self.num_level, 1, kernel_size=1)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, corr, source, target):
        x = corr.clone()

        pos_embed = torch.cat(
            (self.pos_embed_x.repeat(1, 1, self.img_size, 1, 1), self.pos_embed_y.repeat(1, 1, 1, self.img_size, 1)),
            dim=4)
        pos_embed = pos_embed.flatten(2, 3)

        x = torch.cat((x.transpose(-1, -2), target), dim=3) + pos_embed
        x = self.proj(self.blocks(x)).transpose(-1, -2)  # swapping the axis for swapping self-attention.

        '''Level convolutional'''
        channel_map = self.channel_interaction(x)
        x = x * torch.sigmoid(channel_map) + corr

        x = torch.cat((x, source), dim=3) + pos_embed
        x = self.proj(self.blocks(x))

        '''Spatial convolutional'''
        spatial_map = self.spatial_interaction(x)
        x = x * torch.sigmoid(spatial_map) + corr

        return x.mean(1)


class FeatureExtractionHyperPixel(nn.Module):
    def __init__(self, hyperpixel_ids, feature_size, freeze=True):
        super().__init__()
        self.backbone = resnet.resnet101(pretrained=True)
        self.feature_size = feature_size
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        nbottlenecks = [3, 4, 23, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.hyperpixel_ids = hyperpixel_ids

    def forward(self, img):
        r"""Extract desired a list of intermediate features"""

        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                # if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            feats[idx] = F.interpolate(feat, self.feature_size, None, 'bilinear', True)

        return feats


class MIAM(nn.Module):
    def __init__(self,
                 feature_size=16,
                 feature_proj_dim=128,
                 depth=4,
                 num_heads=6,
                 mlp_ratio=4,
                 hyperpixel_ids=[0, 8, 20, 21, 26, 28, 29, 30],
                 freeze=True,
                 patch_size = [8,16,32]):
        super().__init__()
        self.feature_size = feature_size
        self.feature_proj_dim = feature_proj_dim
        self.decoder_embed_dim = self.feature_size ** 2 + self.feature_proj_dim

        channels = [64] + [256] * 3 + [512] * 4 + [1024] * 23 + [2048] * 3

        self.feature_extraction = FeatureExtractionHyperPixel(hyperpixel_ids, feature_size, freeze)
        self.proj = nn.ModuleList([
            nn.Linear(channels[i], self.feature_proj_dim) for i in hyperpixel_ids
        ])

        self.decoder = TransformerAggregator(
            img_size=self.feature_size, embed_dim=self.decoder_embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_hyperpixel=len(hyperpixel_ids),patch_size=patch_size)

        self.l2norm = FeatureL2Norm()

        self.x_normal = np.linspace(-1, 1, self.feature_size)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))
        self.y_normal = np.linspace(-1, 1, self.feature_size)
        self.y_normal = nn.Parameter(torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False))

    def softmax_with_temperature(self, x, beta, d=1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        exp_x = torch.exp(x / beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b, _, h, w = corr.size()

        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1, h, w, h, w)  # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
        x_normal = self.x_normal.expand(b, w)
        x_normal = x_normal.view(b, w, 1, 1)
        grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

        grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
        y_normal = self.y_normal.expand(b, h)
        y_normal = y_normal.view(b, h, 1, 1)
        grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
        return grid_x, grid_y

    def mutual_nn_filter(self, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)

    def corr(self, src, trg):
        return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)

    def forward(self, target, source):
        B, _, H, W = target.size()

        src_feats = self.feature_extraction(source)
        tgt_feats = self.feature_extraction(target)

        corrs = []
        src_feats_proj = []
        tgt_feats_proj = []
        for i, (src, tgt) in enumerate(zip(src_feats, tgt_feats)):
            corr = self.corr(self.l2norm(src), self.l2norm(tgt))
            corrs.append(corr)
            src_feats_proj.append(self.proj[i](src.flatten(2).transpose(-1, -2)))
            tgt_feats_proj.append(self.proj[i](tgt.flatten(2).transpose(-1, -2)))

        src_feats = torch.stack(src_feats_proj, dim=1)
        tgt_feats = torch.stack(tgt_feats_proj, dim=1)
        corr = torch.stack(corrs, dim=1)

        corr = self.mutual_nn_filter(corr)

        refined_corr = self.decoder(corr, src_feats, tgt_feats)

        grid_x, grid_y = self.soft_argmax(refined_corr.view(B, -1, self.feature_size, self.feature_size))

        flow = torch.cat((grid_x, grid_y), dim=1)
        flow = unnormalise_and_convert_mapping_to_flow(flow)

        return flow