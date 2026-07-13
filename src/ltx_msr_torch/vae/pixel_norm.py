# Derived from ComfyUI: https://github.com/comfyanonymous/ComfyUI
# Source file: comfy/ldm/lightricks/vae/pixel_norm.py
# Source commit: dd17debce517f8818ae9910b437cb1ebaa673176
# Modified for ltx-msr-torch on 2026-07-13.
# SPDX-License-Identifier: GPL-3.0-only
import torch
from torch import nn


class PixelNorm(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=self.dim, keepdim=True) + self.eps)
