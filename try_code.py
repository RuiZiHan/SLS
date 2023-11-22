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


def forward_side_adapter(self, x):
    x = self.patch_embed(x)
    if self.absolute_pos_embed is not None:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)

    # x = self.layers(x)
    x_backbone = x
    x_bypass = x
    bypass_input = x
    insert_list = []
    for depth in range(1):
        x_backbone = self.layers[depth](x_backbone)
        insert_list.append(x_backbone)

    bypass_output = []
    for depth in range(1):  # short_bypass
        x_bypass = self.adapt_merging_short[depth](bypass_input)
        x_bypass = self.adapter_short[depth](
            x=self.learnable_gate_short[depth] * x_bypass + (1 - self.learnable_gate_short[depth]) *
              insert_list[depth],
            layer=depth
        )
    bypass_output.append(x_bypass)
    for i in range(2):
        x_backbone = self.layers[1](x=x_backbone + self.insert_scale * bypass_output[0], start_point=i,
                                    end_point=i + 1, bypass_insert=self.insert_scale * bypass_output[0])
    insert_list.append(x_backbone)

    for depth in range(2):  # middle_bypass
        if depth == 0:
            x_bypass = self.adapt_merging[depth](bypass_input)
            x_bypass = self.adapter[depth](
                x=self.learnable_gate[depth] * x_bypass + (1 - self.learnable_gate[depth]) * insert_list[
                    depth], layer=depth)
        else:
            x_bypass = self.adapt_merging[depth](x_bypass)
            x_bypass = self.adapter[depth](
                x=self.learnable_gate[depth] * x_bypass + (1 - self.learnable_gate[depth]) * insert_list[
                    depth], layer=depth)
    bypass_output.append(x_bypass)
    for i in range(18):
        x_backbone = self.layers[2](x=x_backbone + self.insert_scale * bypass_output[1], start_point=i,
                                    end_point=i + 1, bypass_insert=self.insert_scale * bypass_output[1])
    insert_list.append(x_backbone)

    for depth in range(3):  # long_bypass
        if depth == 0:
            x_bypass = self.adapt_merging_long[depth](bypass_input)
            x_bypass = self.adapter_long[depth](
                x=self.learnable_gate_long[depth] * x_bypass + (1 - self.learnable_gate_long[depth]) *
                  insert_list[depth], layer=depth)
        else:
            x_bypass = self.adapt_merging_long[depth](x_bypass)
            x_bypass = self.adapter_long[depth](
                x=self.learnable_gate_long[depth] * x_bypass + (1 - self.learnable_gate_long[depth]) *
                  insert_list[depth], layer=depth)
    bypass_output.append(x_bypass)

    for i in range(2):
        x_backbone = self.layers[3](x=x_backbone + self.insert_scale * bypass_output[2], start_point=i,
                                    end_point=i + 1, bypass_insert=self.insert_scale * bypass_output[2])
    x = x_backbone

    x = self.norm(x)  # B L C
    x = self.avgpool(x.transpose(1, 2))  # B C 1
    x = torch.flatten(x, 1)
    return x


def forward_side_basicLayer(self, x, bypass_insert=None, start_point=0, end_point=None):
    if not end_point:
        end_point = self.depth

    for depth in range(start_point, end_point):  # 0->17
        if bypass_insert is not None:
            x = self.blocks[depth](x, bypass_insert=bypass_insert)
        else:
            x = self.blocks[depth](x)

    # if self.layer_id < - 1:
    #     x = self.last_adapter(x)

    if end_point == self.depth:
        if self.downsample is not None:
            x = self.downsample(x)

    return x


def forward_side_block(self, x, bypass_insert=None):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = self.norm1(x)
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x,
                                                              self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)
    if bypass_insert is not None:
        x = x + bypass_insert

    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s

    return x


def forward_mix_adapter(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = self.norm1(x)
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x,
                                                              self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)

    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s

    return x


def forward_spacial_adapter_vit(self, x):
    # residual = x
    # x = self.norm1(x)
    # attn, feature = self.adapter_attn(x)
    # x = (residual + self.drop_path(self.attn(x)) + feature) * attn
    #
    # residual = x
    # x = self.norm2(x)
    # attn, feature = self.adapter_mlp(x)
    # x = (residual + self.drop_path(self.mlp(x)) + feature) * attn

    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
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


def set_Side_Adapter(model, method, dim=8, s=1, xavier_init=False):
    self = model
    self.adapter_short = nn.ModuleList()
    self.adapt_merging_short = nn.ModuleList()
    self.learnable_gate_short = nn.ParameterList()

    self.adapter = nn.ModuleList()
    self.adapt_merging = nn.ModuleList()
    self.learnable_gate = nn.ParameterList()

    self.adapter_long = nn.ModuleList()
    self.adapt_merging_long = nn.ModuleList()
    self.learnable_gate_long = nn.ParameterList()

    self.insert_scale = s

    middle_dim = 4
    for i in range(1):
        adapter = Adapter(d_model=128 * (2 ** (i + 1)), dropout=0.1,
                          bottleneck=middle_dim,
                          init_option="lora",
                          adapter_scalar=1)
        # adapter = ConvAdapter(input_dim=128 * (2 ** (i + 1)))
        merging = AdaptMerging((56 // (2 ** i), 56 // (2 ** i)), 128 * (2 ** i))
        learnable_gate = nn.Parameter(torch.tensor(0.5))
        self.adapter_short.append(adapter)
        self.adapt_merging_short.append(merging)
        self.learnable_gate_short.append(learnable_gate)
    for i in range(2):
        adapter = Adapter(d_model=128 * (2 ** (i + 1)), dropout=0.1,
                          bottleneck=middle_dim,
                          init_option="lora",
                          adapter_scalar=1)
        # adapter = ConvAdapter(input_dim=128 * (2 ** (i + 1)))
        merging = AdaptMerging((56 // (2 ** i), 56 // (2 ** i)), 128 * (2 ** i))
        learnable_gate = nn.Parameter(torch.tensor(0.5))
        self.adapter.append(adapter)
        self.adapt_merging.append(merging)
        self.learnable_gate.append(learnable_gate)
    for i in range(3):
        adapter = Adapter(d_model=128 * (2 ** (i + 1)), dropout=0.1,
                          bottleneck=middle_dim,
                          init_option="lora",
                          adapter_scalar=1)
        # adapter = ConvAdapter(input_dim=128 * (2 ** (i + 1)))
        merging = AdaptMerging((56 // (2 ** i), 56 // (2 ** i)), 128 * (2 ** i))
        learnable_gate = nn.Parameter(torch.tensor(0.5))
        self.adapter_long.append(adapter)
        self.adapt_merging_long.append(merging)
        self.learnable_gate_long.append(learnable_gate)

    bound_method = forward_side_adapter.__get__(model, model.__class__)
    setattr(model, 'forward_features', bound_method)


def set_layer_forward(model):
    for _ in model.children():
        if type(_) == timm.models.swin_transformer.BasicLayer:
            bound_method = forward_side_basicLayer.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_layer_forward(_)


def set_block_forward(model, s, xavier_init=False):
    for _ in model.children():
        if type(_) == timm.models.swin_transformer.SwinTransformerBlock:
            _.adapter_attn = Convpass_swin(8, xavier_init, _.dim)
            _.adapter_mlp = Convpass_swin(8, xavier_init, _.dim)
            _.s = s
            bound_method = forward_side_block.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_block_forward(_, s, xavier_init)


def set_limited_convpass(model, method, dim=8, s=1, xavier_init=False):
    for i in range(12):
        if i > 3:
            model.blocks[i].adapter_attn = Convpass(dim, xavier_init)
            model.blocks[i].adapter_mlp = Convpass(dim, xavier_init)
            model.blocks[i].s = s
            bound_method = forward_block.__get__(model.blocks[i], model.blocks[i].__class__)
            setattr(model.blocks[i], 'forward', bound_method)


def backbone_weight_fusioin(model):
    for index in range(2, 6):  # 2，3，4，5
        index = 2 * index  # 4,6,8,10
        new_weight1 = OrderedDict()
        new_weight2 = OrderedDict()
        check_point1 = model.blocks[index + 1].state_dict()
        check_point2 = model.blocks[index].state_dict()
        for name, p in check_point1.items():
            new_weight1[name] = p
        for name, p in check_point2.items():
            new_weight2[name] = 0.5 * p + 0.5 * new_weight1[name]
        block = model.blocks[index]
        block.load_state_dict(new_weight2)


class AdaptMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.adapt_reduction = Adapter(d_model=4 * dim, dropout=0.1, bottleneck=8, init_option="lora",
                                       adapter_scalar=1.,
                                       expand=True, up_dim=2 * dim)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.adapt_reduction(x)

        return x


class AdaptExpand(nn.Module):  # Swin-Unet
    """
    块状扩充，尺寸翻倍，通道数减半
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        """
        Args:
            input_resolution: 解码过程的feature map的宽高
            dim: frature map通道数
            dim_scale: 通道数扩充的倍数
            norm_layer: 通道方向归一化
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # 通过全连接层来扩大通道数
        # self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()  # 1024*2048
        self.norm = norm_layer(dim // dim_scale)
        self.adapter = Adapter(d_model=dim, dropout=0.2, bottleneck=4, init_option="lora",
                               adapter_scalar="learnable_scalar",
                               expand=True, up_dim=2 * dim)  # 1024*4+4*2048

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        # 先把通道数翻倍
        # x = self.expand(x)  # + self.adapter(x, add_residual=False)
        x = self.adapter(x, add_residual=False)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # 将各个通道分开，再将所有通道拼成一个feature map
        # 增大了feature map的尺寸
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        # 通道翻倍后再除以4，实际相当于通道数减半
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="learnable_scalar",
                 adapter_layernorm_option="in",
                 expand=False,
                 up_dim=None,
                 model=None
                 ):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()

        if expand:
            self.up_proj = nn.Linear(self.down_size, up_dim)
        else:
            self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                # nn.init.zeros_(self.down_proj.weight)
                nn.init.zeros_(self.up_proj.weight)
                # nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

        self.attn_down = nn.Linear(768, 8)
        self.attn_up = nn.Linear(8, 768)
        self.act = nn.Sigmoid()

    def forward(self, x, add_residual=False, residual=None, layer=0):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        attn = self.act(self.attn_up(self.non_linear_func(self.attn_down(torch.mean(x, dim=1, keepdim=True)))))

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output * attn


class attn_adapter(nn.Module):
    def __init__(self, dim, n, bottleneck=64, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = bottleneck // num_heads
        self.scale = head_dim ** -0.5

        self.down = nn.Linear(n, bottleneck)
        self.act = QuickGELU()
        self.up = nn.Linear(bottleneck, n)
        self.qkv = nn.Linear(bottleneck, bottleneck * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(bottleneck, bottleneck)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = rearrange(x, "B N C->B C N")
        x = self.down(x)
        x = self.act(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.up(x)
        x = rearrange(x, "B C N->B N C")
        return x


class spacial_adapter(nn.Module):
    def __init__(self, n, c, bottleneck, dropout, scale=1.0):
        super().__init__()
        self.dropout = dropout
        self.down = nn.Linear(n, bottleneck)
        self.non_linear_func = QuickGELU()
        self.up = nn.Linear(bottleneck, n)
        self.norm = nn.LayerNorm(n)
        self.scale = scale
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.bias)

    def forward(self, x, add_residual=False, residual=None, layer=0):
        B, N, C = x.shape
        if add_residual:
            residual = x
        else:
            residual = 0
        x = rearrange(x, 'B N C->B C N')
        x = self.norm(x)
        x = self.down(x)
        x = self.non_linear_func(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.up(x)
        x = rearrange(x, 'B C N->B N C')

        return (x + residual) * self.scale


class mixAdapter(nn.Module):
    def __init__(self, n, dim, bottleneck, dropout, channel_weight=1.0, xavier_init=False):
        super().__init__()
        # self.channel_adapter = Adapter(d_model=dim, dropout=dropout, bottleneck=bottleneck,
        #                                init_option="lora",
        #                                adapter_scalar=1,
        #                                adapter_layernorm_option="in", )
        self.channel_adapter = Convpass_swin(bottleneck // 2, xavier_init, dim)
        self.spacial_adapter = spacial_adapter(n=n, c=dim, bottleneck=bottleneck * 8, dropout=dropout)
        self.channel_weight = nn.Parameter(torch.tensor(channel_weight))

        # self.token_down = nn.Linear(n, bottleneck)
        # self.channel_down = nn.Linear(dim, bottleneck)
        # self.token_up = nn.Linear(bottleneck, n)
        # self.channel_up = nn.Linear(bottleneck, dim)
        # self.non_linear_func = nn.ReLU()
        # self.norm1 = nn.LayerNorm(n)
        # with torch.no_grad():
        #     nn.init.kaiming_uniform_(self.token_down.weight, a=math.sqrt(5))
        #     nn.init.zeros_(self.token_up.weight)
        #     nn.init.zeros_(self.token_down.bias)
        #     nn.init.zeros_(self.token_up.bias)
        #
        #     nn.init.kaiming_uniform_(self.channel_down.weight, a=math.sqrt(5))
        #     nn.init.zeros_(self.channel_up.weight)
        #     nn.init.zeros_(self.channel_down.bias)
        #     nn.init.zeros_(self.channel_up.bias)

    def forward(self, x, add_residual=False):
        if add_residual:
            residual = x
        else:
            residual = 0
        x = self.channel_weight * self.channel_adapter(x) + (
                1 - self.channel_weight) * self.spacial_adapter(x, add_residual=False)
        # x = rearrange(x, "B N C->B C N")
        # x = self.norm1(x)
        # x = self.token_down(x)  # B C N/r
        # x = self.non_linear_func(x)
        # x = rearrange(x, "B C N->B N C")  # B N/r C
        # x = self.channel_down(x)  # B N/r C/r
        # x = self.non_linear_func(x)
        # x = self.channel_up(x)  # B N/r C
        # x = self.non_linear_func(x)
        # x = rearrange(x, "B N C->B C N")  # B C N/r
        # x = self.token_up(x)  # B C N
        # x = rearrange(x, "B C N->B N C")

        return x + residual


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=3, padding=1, groups=8)
        # self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        # out = self.pointwise(out)
        return out


class my_adapter_module(nn.Module):  # Se-Bypass
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()
        # 需要有layernorm,且是在降维之前
        self.down1 = nn.Linear(768, 8)
        self.down2 = nn.Linear(768, 8)
        self.act1 = nn.ReLU()
        self.up1 = nn.Linear(8, 768)
        self.up2 = nn.Linear(8, 768)
        self.act2 = nn.Sigmoid()
        self.norm = nn.LayerNorm(768)
        nn.init.kaiming_uniform_(self.down1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up1.weight)
        nn.init.zeros_(self.down1.bias)
        nn.init.zeros_(self.up1.bias)

    def forward(self, x):
        B, N, C = x.shape  # [B,197,768]
        x = self.norm(x)
        feature = self.up1(self.act1(self.down1(x)))
        attn = torch.mean(x, dim=1, keepdim=True)  # [B,1,768]
        attn = self.act2(self.up2(self.act1(self.down2(attn))))
        feature = feature * attn
        return feature


def set_mixAdapter(model, method, dim=8, s=1, xavier_init=False):
    for _ in model.children():
        if type(_) == timm.models.swin_transformer.SwinTransformerBlock:
            _.adapter_attn = Convpass(dim, xavier_init)
            _.adapter_mlp = Adapter(d_model=_.dim, dropout=0.1, bottleneck=8, init_option="lora",
                                    adapter_scalar=1.)
            _.s = s
            bound_method = forward_mix_adapter.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif type(_) == timm.models.vision_transformer.Block:
            _.adapter_attn = Convpass(dim, xavier_init)
            _.adapter_mlp = Convpass(dim, xavier_init)
            _.s = s
            bound_method = forward_spacial_adapter_vit.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_mixAdapter(_, method, dim, s, xavier_init)
