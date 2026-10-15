import torch
import torch.nn as nn
from numpy.ma.core import identity
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from timm.models import create_model

__all__ = ['THT']
tau_thr = 1.5
class Erode(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        return self.pool(x)

class TIM(nn.Module):
    def __init__(self, in_channels=32, TIM_alpha=0.6, kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.interactor = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
        )
        self.in_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')
        self.out_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')
        self.tim_alpha = TIM_alpha

    def forward(self, x, hook=None):
        T, B, num_heads, N, CoH = x.shape
        assert CoH == self.in_channels, f"Expected {self.in_channels} channels, got {CoH}"
        output = []
        x_tim = x[0]
        if hook is not None:
            hook[f"{self._get_name()}_initial"] = x_tim.detach()
        for t in range(T):
            if t == 0:
                output.append(x_tim)
            else:
                x_tim_flat = x_tim.reshape(B * num_heads * N, self.in_channels, 1)
                x_tim_conv = self.interactor(x_tim_flat).reshape(B, num_heads, N, self.in_channels)
                x_tim_in_lif = self.in_lif(x_tim_conv)
                x_tim = x_tim_in_lif * self.tim_alpha + x[t] * (1 - self.tim_alpha)
                x_tim = self.out_lif(x_tim)
                output.append(x_tim)
                if hook is not None and t == T - 1:
                    hook[f"{self._get_name()}_in_lif"] = x_tim_in_lif.detach()
                    hook[f"{self._get_name()}_out_lif"] = x_tim.detach()
        output = torch.stack(output, dim=0)
        return output


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')

        self.c_input = in_features
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        x = self.fc1_lif(x)
        if hook is not None:
            hook[self._get_name() + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()

        if self.c_hidden == self.c_input:
            x = identity + x
            identity = x

        x = self.fc2_lif(x)
        if hook is not None:
            hook[self._get_name() + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, self.c_output, H, W).contiguous()


        x = x + identity
        return x


class TemporalHadamardAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, TIM_alpha=0.6):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.pool = Erode()

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.first_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')
        self.in_channels = dim // num_heads
        self.TIM = TIM(in_channels=self.in_channels, TIM_alpha=TIM_alpha)

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')

        self.talking_heads_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend='cupy')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W
        x = self.first_lif(x)
        if hook is not None:
            hook[self._get_name() + "_first_lif"] = x.detach()

        x = x.flatten(3)
        # T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        if hook is not None:
            hook[self._get_name() + "_q_lif"] = q_conv_out.detach()
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        q = self.TIM(q, hook=hook)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + "_k_lif"] = k_conv_out.detach()
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + "_v_lif"] = v_conv_out.detach()
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + "_kv_before"] = kv.detach()
        kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if hook is not None:
            hook[self._get_name() + "_kv"] = kv.detach()
        x = q.mul(kv)
        x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + "_x_after_kv"] = kv.detach()
        x = x.transpose(3, 4).reshape(T, B, C, N)

        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        
        x = x + identity
        return x


class TemporalHadamardTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TemporalHadamardAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, hook=None):
        x = self.attn(x, hook=hook)
        x = self.mlp(x, hook=hook)
        return x

class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels


        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend="cupy")

        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend="cupy")

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend="cupy")

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend="cupy")

        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=tau_thr, detach_reset=True, backend="cupy")



    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        ratio = 1
        x = self.proj_conv(x.flatten(0, 1))  # have some fire value
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x)
        if hook is not None:
            hook[self._get_name() + "_lif"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        x = self.maxpool(x)
        ratio *= 2

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        x = self.maxpool1(x)
        ratio *= 2

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x)
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        x = self.maxpool2(x)
        ratio *= 2

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.maxpool3(x)
        ratio *= 2

        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())
        if hook is not None:
            hook[self._get_name() + "_lif3"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        return x, (None, None)

class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        patch_embed = SPS(img_size_h=img_size_h,
                              img_size_w=img_size_w,
                              patch_size=patch_size,
                              in_channels=in_channels,
                              embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        block = nn.ModuleList([TemporalHadamardTransformer(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        self.norm = norm_layer(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x, hook=hook)
        for blk in block:
            x = blk(x, hook=hook)
        return x.flatten(3).mean(3)

    def forward(self, x, hook=None):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x, hook=hook)
        x = self.head(x.mean(0))
        return x

@register_model
def THT(pretrained=False, **kwargs):
    model = vit_snn(
        patch_size=16, 
        embed_dims=512, 
        num_heads=16, mlp_ratios=4,
        in_channels=2, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=2, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model