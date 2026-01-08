import torch
import torch.nn as nn
from .vss import VSSBlock


class DualVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0,
        d_state: int = 16,
        **kwargs
    ):
        super().__init__()
        
        assert hidden_dim % 2 == 0, ""
        self.split_dim = hidden_dim // 2
        self.hidden_dim = hidden_dim
        
        self.vss1 = VSSBlock(
            hidden_dim=self.split_dim,
            drop_path=drop_path,
            d_state=d_state,
            **kwargs
        )
        
        self.vss2 = VSSBlock(
            hidden_dim=self.split_dim,
            drop_path=drop_path,
            d_state=d_state,
            **kwargs
        )
    
    def forward(self, x):
        if x.shape[-1] == self.hidden_dim:
            format_hwc = True
            B, H, W, C = x.shape
        else:
            format_hwc = False
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
        
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        out1 = self.vss1(x1)
        out2 = self.vss2(x2)
        
        out = torch.cat([out1, out2], dim=-1)
        
        if not format_hwc:
            out = out.permute(0, 3, 1, 2)
        
        return out


class DualVSSBlockV2(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0,
        d_state: int = 16,
        **kwargs
    ):
        super().__init__()
        
        assert hidden_dim % 2 == 0, "_"
        self.split_dim = hidden_dim // 2
        self.hidden_dim = hidden_dim
        
        self.vss1 = VSSBlock(
            hidden_dim=self.split_dim,
            drop_path=drop_path,
            d_state=d_state,
            **kwargs
        )
        
        self.vss2 = VSSBlock(
            hidden_dim=self.split_dim,
            drop_path=drop_path,
            d_state=d_state,
            **kwargs
        )
        
        self.cross_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        if x.shape[-1] == self.hidden_dim:
            format_hwc = True
            B, H, W, C = x.shape
        else:
            format_hwc = False
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
        
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        out1 = self.vss1(x1)
        out2 = self.vss2(x2)
        
        concat = torch.cat([out1, out2], dim=-1)
        concat_conv = concat.permute(0, 3, 1, 2)
        
        attn_weights = self.cross_attn(concat_conv)
        enhanced = concat_conv * attn_weights
        
        out = self.fusion(enhanced)
        
        if format_hwc:
            out = out.permute(0, 2, 3, 1)
        
        return out