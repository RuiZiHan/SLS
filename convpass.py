import torch
from torch import nn
import timm
import math


def forward_block(self, x):
    # timm==0.4.12
    # x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    # x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s

    # timm==0.9.10
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x)))) + self.drop_path1(
        self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))) + self.drop_path2(
        self.adapter_mlp(self.norm2(x))) * self.s
    return x



def forward_block_attn(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x


def forward_swin_block(self, x):
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


def forward_swin_block_attn(self, x):
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
    x = x + self.drop_path(self.mlp(self.norm2(x)))

    return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class VanillaAdapter(nn.Module):
    def __init__(self, dim, xavier_init):
        super().__init__()
        self.down = nn.Linear(768, 32)
        self.act = QuickGELU()
        self.up = nn.Linear(32, 768)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)

        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)
        # nn.init.kaiming_uniform_(self.adapter_conv.weight, a=math.sqrt(5))
        # nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        self.dim=x_down.shape[-1]

        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)

        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class Convpass_swin(nn.Module):
    def __init__(self, dim=8, xavier_init=True, vit_dim=768):
        super().__init__()
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(vit_dim, dim)
        self.adapter_up = nn.Linear(dim, vit_dim)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.attn_conv = nn.Conv2d(8, 2, 1)
        self.act2 = QuickGELU()
        self.attn_conv_up = nn.Conv2d(2, 8, 1)
        self.nonLinear = nn.Sigmoid()

    def forward(self, x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)

        attn = self.nonLinear(self.attn_conv_up(self.act2(self.attn_conv(self.pooling(x_patch)))))
        # print(attn.shape)
        x_patch = x_patch * attn

        x_patch = self.adapter_conv(x_patch)
        x_down = x_patch.permute(0, 2, 3, 1).reshape(B, -1, self.dim)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up


def set_Convpass(model, method, dim=8, s=1, xavier_init=False):
    if method == 'convpass':
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.adapter_attn = Convpass(dim, xavier_init)
                _.adapter_mlp = Convpass(dim, xavier_init)
                _.s = s
                bound_method = forward_block.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                _.adapter_attn = Convpass_swin(dim, xavier_init, _.dim)
                _.adapter_mlp = Convpass_swin(dim, xavier_init, _.dim)
                _.s = s
                bound_method = forward_swin_block.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_Convpass(_, method, dim, s, xavier_init)
    else:
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block:
                _.adapter_attn = Convpass(dim, xavier_init)
                _.s = s
                bound_method = forward_block_attn.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                _.adapter_attn = Convpass_swin(dim, xavier_init, _.dim)
                _.s = s
                bound_method = forward_swin_block_attn.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_Convpass(_, method, dim, s, xavier_init)


def new_set_Convpass(model, method, dim=8, s=1, xavier_init=False):
    block_count = 0
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Block:
            block_count += 1
            print(block_count)
            _.adapter_attn = Convpass(dim, xavier_init)
            _.adapter_mlp = Convpass(dim, xavier_init)
            _.s = s * (block_count / 11)
            bound_method = forward_block.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)

        elif len(list(_.children())) != 0:
            new_set_Convpass(_, method, dim, s, xavier_init)
