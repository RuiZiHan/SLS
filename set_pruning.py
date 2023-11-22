import timm
import torch
from torch.optim import AdamW
from torch.nn import functional as F
import torch.nn as nn
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model, checkpoint_seq
from timm.models.vision_transformer import Block
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import *
import math
from convpass import Convpass, forward_block, VanillaAdapter, Convpass_swin, QuickGELU
from collections import OrderedDict
from einops import rearrange
from timm.models.layers import Mlp
from timm.models.vision_transformer import Attention


def forward_drop(self, x):
    # timm==0.4.12
    # x = self.patch_embed(x)
    # cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    # if self.dist_token is None:
    #     x = torch.cat((cls_token, x), dim=1)
    # else:
    #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    # x = self.pos_drop(x + self.pos_embed)
    #
    # for i in range(self.current_depth):
    #     x = self.blocks[i](x)
    #
    # x = self.norm(x)
    # if self.dist_token is None:
    #     return self.pre_logits(x[:, 0])
    # else:
    #     return x[:, 0], x[:, 1]

    # timm==0.9.10
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)

    for i in range(self.current_depth):
        x = self.blocks[i](x)

    x = self.norm(x)
    return x


def swin_forward_drop(self, x):
    x = self.patch_embed(x)
    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)

    if self.current_depth >= 23:  # 1->24 2,2,18,2
        # stage4
        for index in range(3):  # stage1,2,3
            x = self.layers[index](x)
        layer_depth = self.current_depth - 22
        x = self.layers[3](x, layer_depth)  # stage4

    elif 22 >= self.current_depth >= 5:
        # stage3
        for index in range(2):  # stage1,2
            x = self.layers[index](x)
        layer_depth = self.current_depth - 4
        x = self.layers[2](x, layer_depth)  # stage3

    x = self.norm(x)  # B L C
    x = self.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)
    return x


def swin_layer_drop(self, x, layer_depth=None):
    if layer_depth is None:
        forward_depth = self.depth
    else:
        forward_depth = layer_depth
    for blk in range(forward_depth):
        x = self.blocks[blk](x)
    # if self.downsample is not None and layer_depth is None:
    #     x = self.downsample(x)
    if self.downsample is not None:
        x = self.downsample(x)
    return x


def set_drop_backbone(model, current_depth=12):
    print(type(model))
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.current_depth = current_depth
        bound_method = forward_drop.__get__(model, model.__class__)
        setattr(model, 'forward_features', bound_method)
    elif type(model) == timm.models.swin_transformer.SwinTransformer:
        model.current_depth = current_depth
        bound_method = swin_forward_drop.__get__(model, model.__class__)
        setattr(model, 'forward_features', bound_method)
    for _ in model.children():
        if type(_) == timm.models.swin_transformer.SwinTransformerStage:
            bound_method = swin_layer_drop.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_drop_backbone(_)
