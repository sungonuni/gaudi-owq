# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
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
from custom_func import CustomFunc

def test_custom_relu_op_function():
    print(torch.ops.custom_op.custom_func)
    input = torch.randn(3, 5, requires_grad=True)
    input_hpu = input.to('hpu').detach()
    input_hpu.requires_grad = True
    
    relu = torch.nn.ReLU(inplace=False)
    output_cpu = relu(input)
    
    custom_relu = CustomFunc()
    output_hpu = custom_relu(input_hpu);
    print(torch.equal(output_hpu.detach().cpu(), output_cpu.detach()))

test_custom_relu_op_function()

