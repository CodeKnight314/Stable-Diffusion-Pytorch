import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 
from typing import Union

class MSA(nn.Module):
    def __init__(self, d_model : int, head : int, dropout : float = 0.3):
        assert d_model % head == 0, f"[Error] d_model {d_model} is not divisible by head {head} in MSA Module"
        super().__init__()

        self.d_model = d_model
        self.head = head
        self.d_k = d_model // head

        self.dropout = nn.Dropout(dropout)

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.head)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.head)
        self.W_V = nn.Linear(self.d_model, self.d_k * self.head)
        self.W_O = nn.Linear(self.d_k * self.head, self.d_model)

    def scaled_dot_product(self, Queries, Keys, Values, Mask : Union[None, torch.Tensor] = None):
        """
        Computes the scaled dot-product attention over the inputs.

        Args:
            Queries, Keys, Values (torch.Tensor): The query, key, and value tensors after being processed through
                their respective linear transformations.
            Mask (torch.Tensor, optional): Optional mask tensor to zero out certain positions in the attention scores.

        Returns:
            torch.Tensor: The output tensor after applying attention and weighted sum operations.
        """
        attn_scores = torch.matmul(Queries, torch.transpose(Keys, -2, -1)) / math.sqrt(self.d_k) # Measures similarities between each set of queries and keys
        if Mask:
            attn_scores = attn_scores.masked_fill(Mask == 0, -1e9)
        QK_probs = torch.softmax(attn_scores, dim = -1) # Scales the similarities between each query in Q to the entire set of Keys as probabilities
        QK_probs = self.dropout(QK_probs)
        output = torch.matmul(QK_probs, Values) # Transforms values into weighted sums, reflecting importance of each value within Values
        return output

    def forward(self, x, Mask : Union[None, torch.Tensor] = None):
        """
        Forward pass of the MSA module. Applies self-attention individually to each head, concatenates the results,
        and then projects the concatenated output back to the original dimensionality.

        Args:
            x (torch.Tensor): Inputs to the self-attention mechanism.
            Mask (torch.Tensor, optional): Optional mask to apply during the attention mechanism.

        Returns:
            torch.Tensor: The output of the MSA module after processing through the attention mechanism and linear output layer.
        """
        Queries = x
        Keys = x
        Values = x
        
        batch_size = Queries.size(0)

        Q = self.W_Q(Queries).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        K = self.W_K(Keys).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        V = self.W_V(Values).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        scaled_values = self.scaled_dot_product(Queries=Q, Keys=K, Values=V, Mask=Mask).transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.head)

        context = self.dropout(self.W_O(scaled_values))

        return context

class VAE_ResidualBlock(nn.Module): 
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        
        self.group_one = nn.GroupNorm(32, input_channels)
        self.conv_one = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        
        self.group_two = nn.GroupNorm(32, output_channels)
        self.conv_two = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        
        if input_channels == output_channels: 
            self.residual = nn.Identity() 
        else: 
            self.residual = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor): 
        residual = x
        x = self.group_one(x)
        x = F.silu(x)
        x = self.conv_one(x)
        x = self.group_two(x)
        x = F.silu(x)
        x = self.conv_two(x)
        
        return x + self.residual(residual)
        
class VAE_AttentionBlock(nn.Module): 
    def __init__(self, input_channels: int): 
        super().__init__()
        
        self.group = nn.GroupNorm(32, input_channels)
        self.msa = MSA(d_model=input_channels, head=1)
        
    def forward(self, x: torch.Tensor): 
        B, C, H, W = x.shape 
        
        residual = x 
        x = self.group(x)
        
        x = x.view((B, C, H*W))
        x = x.transpose(-1, -2)
        x = self.msa(x)
        
        x = x.transpose(-1, -2)
        x = x.view((B, C, H, W))
        x = x + residual
        
        return x
    
class VAE_Encoder(nn.Module): 
    def __init__(self, input_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1), 
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            
            VAE_ResidualBlock(128, 256), 
            VAE_ResidualBlock(256, 256),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            
            VAE_ResidualBlock(256, 512), 
            VAE_ResidualBlock(512, 512),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512), 
            
            VAE_ResidualBlock(512, 512),
            
            nn.GroupNorm(32, 512), 
            nn.SiLU(), 
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor): 
        x = self.conv(x)
        
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        std = variance.sqrt()
        
        x = mean + std * noise
        x *= 0.18215 
        
        return x
        
class VAE_Decoder(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0), 
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),
            VAE_ResidualBlock(512, 512), 
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), 
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), 
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        )
        
        
    def forward(self, x: torch.Tensor): 
        x /= 0.18215
        return self.conv(x)