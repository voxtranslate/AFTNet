import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from torch.nn.parameter import Parameter

class AdaptiveFeatureNorm(nn.Module):
    """Novel Adaptive Feature Normalization module with learnable statistics"""
    def __init__(self, num_features, eps=1e-5):
        super(AdaptiveFeatureNorm, self).__init__()
        self.eps   = eps
        self.gamma = Parameter(torch.ones(1, num_features, 1, 1))
        self.beta  = Parameter(torch.zeros(1, num_features, 1, 1))
        
        # Adaptive statistics network
        self.stats_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features//4, 1),
            nn.ReLU(True),
            nn.Conv2d(num_features//4, num_features*2, 1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Calculate adaptive statistics
        stats = self.stats_net(x)
        adaptive_gamma, adaptive_beta = torch.chunk(stats, 2, dim=1)
        
        # Instance normalization
        var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        
        # Apply adaptive scaling and shifting
        return (1 + adaptive_gamma) * self.gamma * x_norm + adaptive_beta * self.beta

class MultiScaleFrequencyAttention(nn.Module):
    """Novel multi-scale frequency attention module"""
    def __init__(self, dim, num_heads=8):
        super(MultiScaleFrequencyAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        
        self.qkv  = nn.Conv2d(dim, dim*3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        
        # Frequency decomposition branches
        self.freq_decomp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim//4, 3, padding=1, groups=dim//4),
                nn.GELU(),
                nn.Conv2d(dim//4, dim, 1)
            ) for _ in range(3)  # Low, mid, high frequencies
        ])
        
        # Frequency attention weights
        self.freq_weights = nn.Parameter(torch.ones(3))
        self.softmax      = nn.Softmax(dim=0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Multi-head attention
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H, W)
        q, k, v = qkv.unbind(1)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Frequency decomposition
        freq_components = [decomp(x) for decomp in self.freq_decomp]
        freq_weights = self.softmax(self.freq_weights)
        
        # Combine frequency components
        freq_out = sum([w * f for w, f in zip(freq_weights, freq_components)])
        
        # Combine with spatial attention
        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        
        return x + freq_out

class TemporalConsistencyModule(nn.Module):
    """Novel temporal consistency module with adaptive feature alignment"""
    def __init__(self, dim):
        super(TemporalConsistencyModule, self).__init__()
        
        # Feature alignment network
        self.alignment_net = nn.Sequential(
            nn.Conv2d(dim*2, dim//2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim//2, 2, 3, padding=1)
        )
        
        # Temporal attention
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(dim*2, dim//2, 1),
            nn.ReLU(True),
            nn.Conv2d(dim//2, dim, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Conv2d(dim*2, dim, 1)

    def forward(self, current, previous):
        # Calculate optical flow
        flow = self.alignment_net(torch.cat([current, previous], dim=1))
        
        # Warp previous features
        grid = self.get_grid(flow)
        warped_prev = F.grid_sample(previous, grid, align_corners=True)
        
        # Temporal attention
        attn = self.temporal_attn(torch.cat([current, warped_prev], dim=1))
        
        # Feature fusion
        fused = self.fusion(torch.cat([current * attn, warped_prev * (1-attn)], dim=1))
        return fused.float()

    def get_grid(self, flow):
        B, _, H, W = flow.size()
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float().to(flow.device)
        vgrid = grid + flow
        
        # Scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0
        return vgrid.permute(0,2,3,1)

class AdaptiveResidualBlock(nn.Module):
    """Novel adaptive residual block with dynamic routing"""
    def __init__(self, dim):
        super(AdaptiveResidualBlock, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim//4, 3, padding=1),
            AdaptiveFeatureNorm(dim//4),
            nn.GELU(),
            nn.Conv2d(dim//4, dim, 3, padding=1),
            AdaptiveFeatureNorm(dim)
        )
        
        self.branch2 = MultiScaleFrequencyAttention(dim)
        
        # Dynamic routing network
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        route_weights = self.router(x)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        return x + route_weights[:,0:1,:,:] * out1 + route_weights[:,1:2,:,:] * out2

def adjust(x1, x2):
    x1 = F.interpolate(x1, size=x2.shape[2:], mode='nearest')
    return x1

class AFTNet(nn.Module):
    """Advanced Adaptive Frequency-Temporal Network for Image Deblurring"""
    def __init__(self, in_channels=3, dim=32, num_blocks=4):
        super(AFTNet, self).__init__()
        
        # Initial feature extraction
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1),
            AdaptiveFeatureNorm(dim)
        )
        
        # Encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                AdaptiveResidualBlock(dim * (2**i)),
                nn.Conv2d(dim * (2**i), dim * (2**(i+1)), 2, stride=2),
                AdaptiveFeatureNorm(dim * (2**(i+1)))
            ) for i in range(3)
        ])

        # bringing out prev feature to same level as the middle
        self.conv_middle = nn.Conv2d(dim, dim*8, 1)
        
        # Middle blocks with temporal consistency
        self.middle = nn.ModuleList([
            nn.Sequential(
                AdaptiveResidualBlock(dim * 8),
                TemporalConsistencyModule(dim * 8)
            ) for _ in range(num_blocks)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(dim * (2**(i+1)), dim * (2**i), 2, stride=2),
                AdaptiveFeatureNorm(dim * (2**i)),
                AdaptiveResidualBlock(dim * (2**i))
            ) for i in range(2, -1, -1)
        ])
        
        # Feature pyramid fusion
        self.pyramid_fusion = nn.ModuleList([
            nn.Conv2d(dim * (2**i) * 2, dim * (2**i), 1)
            for i in range(3)
        ])

        # Multi-scale output
        self.output_layers = nn.ModuleList([
            nn.Conv2d(dim * (2**i), in_channels, 3, padding=1)
            for i in range(4)
        ])

    def forward(self, x, prev_frame=None):
        r = x
        if prev_frame is None:
            prev_frame = x
            
        # Initial features
        x = self.init_conv(x)
        with torch.no_grad():
            prev_features = self.init_conv(prev_frame)
        
        # Encoder
        encoder_features = [x]
        for enc in self.encoder:
            x = enc(x)
            encoder_features.append(x)

        prev_features = F.interpolate(prev_features, size=x.shape[2:], mode='bicubic', align_corners=False)
        prev_features = self.conv_middle(prev_features)

        # Middle blocks with temporal consistency
        for block in self.middle:
            x = block[0](x)                 # Residual block
            x = block[1](x, prev_features)  # Temporal consistency
            prev_features = x

        # Multi-scale outputs
        outputs = [(self.output_layers[-1](x)+F.interpolate(r, size=x.shape[2:], mode='bicubic', align_corners=False)).clamp(0, 1)] 
        
        # Decoder with feature pyramid fusion
        for i, dec in enumerate(self.decoder):
            # Upsample current features
            x = dec[0](x)  # Upsample
            
            # Fusion with encoder features
            s = encoder_features[::-1][1:][i]
            x = adjust(x, s)
            x = self.pyramid_fusion[::-1][i](torch.cat([x, s], dim=1))
            
            # Apply remaining decoder operations
            x = dec[1:](x).float()
            
            # Generate output at current scale
            outputs.append((self.output_layers[-(i+2)](x)+F.interpolate(r, size=x.shape[2:], mode='bicubic', align_corners=False)).clamp(0, 1))
        
        return outputs[::-1]  # Return multi-scale outputs from fine to coarse
