# -*- coding: utf-8 -*-
# @Author  : Lintao Peng
# @File    : SGFMT.py
# coding=utf-8
# Design based on the Vit

import torch
from IntmdSequential import IntermediateSequential


#实现了自注意力机制，相当于unet的bottleneck层
class SelfAttention(torch.nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super(SelfAttention,self).__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(dropout_rate)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super(Residual,self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm,self).__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(torch.nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super(PreNormDrop,self).__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super(FeedForward,self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        dim,  #512
        depth,  #4
        heads,  #8
        mlp_dim,  #4096
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super(TransformerModel,self).__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)

    def forward(self, x):
        return self.net(x)