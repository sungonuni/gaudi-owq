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
import os

os.environ['LOG_LEVEL_ALL'] = '4' 
os.environ['ENABLE_CONSOLE'] = 'true'

def dequant_cpu(input_tensor, step_size, zero_scale):
    return (input_tensor * step_size) + zero_scale

def test_custom_dequant():
    print(torch.ops.custom_op.custom_func)

    input_a = torch.randint(-8, 7, (64, 64), dtype=torch.int8)
    input_a_hpu = input_a.to('hpu').detach()

    step_size = 0.38
    zero_scale = 0.173

    dequant_a_cpu = dequant_cpu(input_a, step_size, zero_scale)

    custom_dequant = CustomFunc()
    dequant_a_hpu = custom_dequant(input_a_hpu.to(torch.float), step_size, zero_scale)
    print(torch.equal(dequant_a_hpu.detach().cpu(), dequant_a_cpu.detach()))

test_custom_dequant()

"""
def test_custom_add():
    print(torch.ops.custom_op.custom_func)

    input_a = torch.randn(64, 64, requires_grad=True)
    input_a_hpu = input_a.to('hpu').detach()

    input_b = torch.randn(64, 64, requires_grad=True)
    input_b_hpu = input_b.to('hpu').detach()

    output_cpu = input_a + input_b

    custom_add = CustomFunc()
    output_hpu = custom_add(input_a_hpu, input_b_hpu);
    print(torch.equal(output_hpu.detach().cpu(), output_cpu.detach()))

test_custom_add()
"""

"""
def test_custom_relu_op_function():
    print(torch.ops.custom_op.custom_func)
    input = torch.randn(3, 5, requires_grad=True)
    input_hpu = input.to('hpu').detach()
    input_hpu.requires_grad = True
    relu = torch.nn.ReLU(inplace=False)
    output_cpu = relu(input)
    out = torch.ones_like(output_cpu)
    out_hpu = out.to('hpu')

    custom_relu = CustomFunc()
    output_hpu = custom_relu(input_hpu);

    print(torch.equal(output_hpu.detach().cpu(), output_cpu.detach()))

test_custom_relu_op_function()
"""