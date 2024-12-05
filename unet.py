import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Union
import math

class UNet_MSA(nn.Module):
    def __init__(self, d_model: int, head: int, dropout: float = 0.3):
        assert d_model % head == 0, f"[Error] d_model {d_model} is not divisible by head {head} in MSA Module"
        super().__init__()

        self.d_model = d_model
        self.head = head
        self.d_k = d_model // head

        self.dropout = nn.Dropout(dropout)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, Q, K, V, mask: Union[None, torch.Tensor] = None):
        batch_size = Q.size(0)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, Q, K, V, mask: Union[None, torch.Tensor] = None):
        batch_size = Q.size(0)

        # Linear projections and reshape
        Q = self.W_Q(Q).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        attn_output = self.scaled_dot_product(Q, K, V, mask)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(attn_output)
        output = self.dropout(output)

        return output

class UNet_attention(nn.Module): 
    def __init__(self, channels: int, heads: int):
        super().__init__() 
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.context_proj = nn.Linear(768, channels)
        
        self.layer_norm_1 = nn.LayerNorm(channels)
        self.msa = UNet_MSA(channels, heads)
        
        self.layer_norm_2 = nn.LayerNorm(channels)
        self.csa = UNet_MSA(channels, heads)
        
        self.layer_norm_3 = nn.LayerNorm(channels)
        self.linear_1 = nn.Linear(channels, channels * 4 * 2)
        self.linear_2 = nn.Linear(channels * 4, channels)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor): 
        B, C, H, W = x.shape
        
        context = self.context_proj(context)
        
        residual = x 
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        x = x.view((B, C, H * W)).transpose(-1, -2)
        
        x = self.layer_norm_1(x)
        x = x + self.msa(x, x, x)
        
        x = self.layer_norm_2(x)
        x = x + self.csa(x, context, context)
        
        x = self.layer_norm_3(x)
        ff_output = self.linear_1(x)
        x, gate = ff_output.chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_2(x)
        
        x = x.transpose(-1, -2).view((B, C, H, W))
        
        return x + residual

class UNet_residual(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, time_channels: int): 
        super().__init__() 
        
        self.time_embedding = nn.Linear(time_channels, output_channels)
        
        self.conv_one = nn.Sequential(
            nn.GroupNorm(32, input_channels), 
            nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv_two = nn.Sequential(
            nn.GroupNorm(32, output_channels), 
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        )
        
        if input_channels == output_channels:
            self.residual_layer = nn.Identity() 
        else: 
            self.residual_layer = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
            
    def forward(self, x: torch.Tensor, time: torch.Tensor): 
        residual = x 
        x = self.conv_one(x)
        
        time = F.silu(time)
        time_embed = self.time_embedding(time)
        
        x = x + time_embed[:, :, None, None]
        
        x = self.conv_two(x)
        x = x + self.residual_layer(residual)
        return x
    
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)
    
class Downsample(nn.Module): 
    def __init__(self, channels: int): 
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor):
        return self.conv(x)
    
class UNet(nn.Module): 
    def __init__(self, input_channels: int = 4, hidden_channels: int = 320, time_channels: int = 320, output_channels: int = 4, num_blocks: int = 11): 
        super().__init__() 
        
        channels = hidden_channels

        self.input_conv = nn.Conv2d(input_channels, 320, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Conv2d(320, output_channels, kernel_size=3, padding=1)

        self.downsample_branch = nn.ModuleList()
        for i in range(num_blocks): 
            if i == 0:
                self.downsample_branch.append(UNet_residual(channels, channels, time_channels))
                self.downsample_branch.append(UNet_attention(channels, 8))
                self.downsample_branch.append(UNet_residual(channels, channels, time_channels))
                self.downsample_branch.append(UNet_attention(channels, 8))
            elif (i+1) % 3 == 0 and (i+1) != num_blocks - 1: 
                self.downsample_branch.append(Upsample(channels//2))
                channels = channels * 2
                self.downsample_branch.append(UNet_residual(channels//2, channels, time_channels))
                self.downsample_branch.append(UNet_attention(channels, 8))
                self.downsample_branch.append(UNet_residual(channels, channels, time_channels))
                self.downsample_branch.append(UNet_attention(channels, 8))
            elif i == num_blocks - 1: 
                self.downsample_branch.append(UNet_residual(channels, channels, time_channels))
                self.downsample_branch.append(UNet_residual(channels, channels, time_channels))
                
        self.bottleneck = nn.Sequential(
            UNet_residual(channels, channels, time_channels), 
            UNet_attention(channels, 8), 
            UNet_residual(channels, channels, time_channels)
        )
        
        self.upsample_branch = nn.ModuleList()
        for i in range(num_blocks): 
            if i == 0:
                self.upsample_branch.append(UNet_residual(channels, channels, time_channels))
                self.upsample_branch.append(UNet_residual(channels, channels, time_channels))
            elif (i+1) % 3 == 0 and (i+1) != num_blocks - 1: 
                self.upsample_branch.append(Upsample(channels//2))
                channels = channels // 2
                self.upsample_branch.append(UNet_residual(channels*2, channels, time_channels))
                self.upsample_branch.append(UNet_attention(channels, 8))
                self.upsample_branch.append(UNet_residual(channels, channels, time_channels))
                self.upsample_branch.append(UNet_attention(channels, 8))
            elif i == num_blocks - 1: 
                self.upsample_branch.append(UNet_residual(channels, channels, time_channels))
                self.upsample_branch.append(UNet_attention(channels, 8))
                self.upsample_branch.append(UNet_residual(channels, channels, time_channels))
                self.upsample_branch.append(UNet_attention(channels, 8))
                
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        x = self.input_conv(x)
        skip_connections = []
        
        for layer in self.downsample_branch:
            if isinstance(layer, (UNet_residual, UNet_attention)):
                x = layer(x, time) if isinstance(layer, UNet_residual) else layer(x, context)
            else:
                x = layer(x)
            skip_connections.append(x)
            
        x = self.bottleneck(x)
        
        for layer in self.upsample_branch:
            if skip_connections:
                x = torch.cat([x, skip_connections.pop()], dim=1)
            if isinstance(layer, (UNet_residual, UNet_attention)):
                x = layer(x, time) if isinstance(layer, UNet_residual) else layer(x, context)
            else:
                x = layer(x)
                
        return self.output_conv(x)

def test_unet_forward():
    unet = UNet()

    batch_size = 2
    height = 64
    width = 64

    x = torch.randn(batch_size, 4, height, width)
    context = torch.randn(batch_size, 77, 768)
    time = torch.randn(batch_size, 320)
    output = unet(x, context, time)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")
       
  
if __name__ == "__main__":
   test_unet_forward()