# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: the underlying kernel comes straight from the Triton tutorials
# see https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py


import torch
import torch.nn as nn

from xformers.components.residual import LayerNormStyle
from xformers.triton.k_residual_ln import _ResidualLayerNorm


class ResidualLayerNorm(nn.Module):
    """
    Handle a layer normalization and the residual path in a single layer.
    """

    def __init__(
        self, layer_norm_style: LayerNormStyle, normalized_shape: int, eps=1e-05
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.epsilon = eps
        self.layer_norm_style = layer_norm_style

    def forward(self, x, y):
        return _ResidualLayerNorm.apply(
            x, y, self.weight, self.bias, self.epsilon, self.layer_norm_style
        )
