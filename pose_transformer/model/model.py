import math
import torch
import torch.nn as nn
from functools import partial
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.U_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.U_msa = nn.Linear(dim, dim)

    def forward(self, x, weight):
        B, N, D = x.shape

        # Easy way to get [q,k,v]
        qkv = self.U_qkv(x)
        # Apply weight filtering to exclude invalid kpts
        if weight is not None:
            qkv = weight[:,:,None] * qkv

        # TODO: Extract q, k, v  ~ (B, num_head, N, D_h) where D_h = D / num_heads
        # qkv shape: (B, N, D*3) -> reshape to (B, N, 3, num_heads, D_h) -> permute to (3, B, num_heads, N, D_h)
        qkv = qkv.reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, N, D_h)

        # TODO: Compute multihead self-attention map A ~ (B, num_head, N, N)
        # A = softmax(q @ k^T / sqrt(D_h))
        A = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        A = A.softmax(dim=-1)

        # TODO: MSA forward; multiplying attention map A with value v
        output = A @ v  # (B, num_heads, N, D_h)

        # TODO: Concatenate multihead output and perfrom linear projection to get output
        # Transpose and reshape to (B, N, D)
        output = output.transpose(1, 2).reshape(B, N, D)
        output = self.U_msa(output)

        return output


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Multihead attention block
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

        # Layer normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, weight):
        # TODO: Implement Encoder, consists of MSA and mlp block
        # Residual connection with layer norm: x = x + attn(norm1(x))
        x = x + self.attn(self.norm1(x), weight)
        # Residual connection with layer norm: x = x + mlp(norm2(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PoseTransformer(nn.Module):
    def __init__(self, num_joints=21, in_chans=2, embed_dim=32, depth=4, num_heads=8, 
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm):
        """ 
        Args:
            num_joints (int): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        # Patch embedding and positional embedding 
        self.patch_embed = nn.Linear(in_chans, embed_dim)
        self.pos_encod = nn.Parameter(torch.zeros(1, num_joints, embed_dim))

        # Transformer Encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer)
            for _ in range(depth)])

        # MLP head to project embedding vectors to 3D joints
        self.mlp_head = nn.Sequential(
            norm_layer(embed_dim),
            nn.Linear(embed_dim , 3)
        )

        # Initialize weight
        trunc_normal_(self.pos_encod, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def prepare_tokens(self, x):
        # TODO: Linearly project 2d kpts to embedding vector with self.patch_embed
        x = self.patch_embed(x)
        # TODO: Add positional encoding self.pos_encod
        x = x + self.pos_encod
        return x


    def forward(self, x, weight):
        # Patch embedding of input 2D hand kpts
        x = self.prepare_tokens(x)

        # Pass through transformer encoder blocks
        for blk in self.blocks:
            x = blk(x, weight)

        # TODO: pass output from last encoder block to MLP head
        x = self.mlp_head(x)
        return x