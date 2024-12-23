# ******************************************************************************
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import torch
import os
import habana_frameworks.torch.core
from pathlib import Path

my_dir = os.path.realpath(__file__)
my_len = my_dir.rfind("/")
base_dir = my_dir[:my_len]

custom_func_op_lib_path = str(
    next(
        Path(
            next(Path(os.path.join(base_dir, "build")).glob("lib.linux-x86_64-*"))
        ).glob("hpu_custom_func.cpython-*-x86_64-linux-gnu.so")
    )
)
torch.ops.load_library(custom_func_op_lib_path)


class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_a, input_b):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor = torch.ops.custom_op.custom_func(input_a, input_b)
        ctx.tensor = tensor
        return tensor


class CustomFunc(torch.nn.Module):
    def __init__(self):
        super(CustomFunc, self).__init__()

    def forward(self, input_a, input_b):
        return CustomFunction.apply(input_a, input_b)

    def extra_repr(self):
        return "CustomFunc for float32 only"
