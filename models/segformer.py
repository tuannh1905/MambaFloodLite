import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(_init_weights)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention, self).__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.apply(_init_weights)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class LinearMLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(LinearMLP, self).__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerModel(nn.Module):
    # ✓ ĐÃ SỬA: Đổi img_size thành input_size để khớp chuẩn project
    def __init__(self, in_channels=3, num_classes=1,
                 embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8],
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_path_rate=0.1,
                 decoder_embed_dim=256,
                 input_size=256):

        super(SegFormerModel, self).__init__()

        self.num_classes = num_classes
        
        # ✓ KIỂM TRA TOÁN HỌC: Mạng downsample 4 lần (stride = 4, 2, 2, 2 -> 32)
        if input_size % 32 != 0:
            raise ValueError(f"SegFormer yêu cầu input_size chia hết cho 32. Kích thước {input_size} không hợp lệ.")

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = OverlapPatchEmbed(img_size=input_size,       patch_size=7, stride=4,
                                              in_chans=in_channels,    embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=input_size // 4,  patch_size=3, stride=2,
                                              in_chans=embed_dims[0],  embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=input_size // 8,  patch_size=3, stride=2,
                                              in_chans=embed_dims[1],  embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=input_size // 16, patch_size=3, stride=2,
                                              in_chans=embed_dims[2],  embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList([Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias=True,
                                           drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0])
                                     for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias=True,
                                           drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1])
                                     for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias=True,
                                           drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2])
                                     for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias=True,
                                           drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3])
                                     for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.linear_c4 = LinearMLP(embed_dims[3], decoder_embed_dim)
        self.linear_c3 = LinearMLP(embed_dims[2], decoder_embed_dim)
        self.linear_c2 = LinearMLP(embed_dims[1], decoder_embed_dim)
        self.linear_c1 = LinearMLP(embed_dims[0], decoder_embed_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_embed_dim * 4, decoder_embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_embed_dim),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(decoder_embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]

        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        c1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed2(c1)
        for blk in self.block2:
            x = blk(x, H, W)
        c2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed3(c2)
        for blk in self.block3:
            x = blk(x, H, W)
        c3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed4(c3)
        for blk in self.block4:
            x = blk(x, H, W)
        c4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        target_size = c1.shape[2:]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(B, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=target_size, mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(B, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=target_size, mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(B, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=target_size, mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(B, -1, c1.shape[2], c1.shape[3])

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(x)
        x = self.linear_pred(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x

# ✓ ĐÃ SỬA: Hàm build_model nhận thêm input_size
def build_model(num_classes=1, input_size=256):
    return SegFormerModel(
        in_channels=3,
        num_classes=num_classes,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.1,
        decoder_embed_dim=256,
        input_size=input_size, 
    )