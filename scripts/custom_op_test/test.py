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

def test_custom_func_op_function():
    print(torch.ops.custom_op.custom_func)

    input_a = torch.randn(16, 16, requires_grad=True)
    input_a_hpu = input_a.to('hpu').detach()

    input_b = torch.randn(16, 16, requires_grad=True)
    input_b_hpu = input_b.to('hpu').detach()

    output_cpu = input_a + input_b
    
    custom_add = CustomFunc()
    output_hpu = custom_add(input_a_hpu, input_b_hpu);
    print(torch.equal(output_hpu.detach().cpu(), output_cpu.detach()))

test_custom_func_op_function()

