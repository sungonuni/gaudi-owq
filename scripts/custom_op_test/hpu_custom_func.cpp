/*******************************************************************************
 * Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

/*
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>

bool register_custom_func() {
    // Registering custom_op::custom_add
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // user param callback
    // auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
    //   HPU_PARAMS_STUB(ns_ReluKernel::Params);
    //   params->threshold.f = 0.0;
    //   return params;
    // };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_func", //schema name
        "custom_gemv_deq_int8_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        nullptr);
    std::cout << "cpp registered custom_op::custom_func\n";
    return true;
}

at::Tensor custom_func_execute(
    torch::Tensor input_a, torch::Tensor input_b) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_func();
  TORCH_CHECK(registered, "custom_func kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_func");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}


TORCH_LIBRARY(custom_op, m) {
  m.def("custom_func(Tensor input_a, Tensor input_b) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_func", custom_func_execute);
}
*/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>
typedef struct sParam{
    float step_size;
    float zero_point;
}gemvParam; // kernel scalar param list (editable)

bool register_custom_func() {
    // Registering custom_op::custom_func
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::SCALAR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::SCALAR, 2};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // user param callback
    // kernel scalar param list (editable)
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(gemvParam);
      params->step_size = inputs[1].to<float>();
      params->zero_point = inputs[2].to<float>();
      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::custom_func", //schema name
        "custom_gemv_deq_int8_gaudi2", // guid
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::custom_relu\n";
    return true;
}

at::Tensor custom_func_execute(
    torch::Tensor input_a, c10::Scalar step_size, c10::Scalar zero_scale) {
  TORCH_CHECK(input_a.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_func();
  TORCH_CHECK(registered, "custom_func kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, step_size, zero_scale};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_func");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_func(Tensor self, Scalar step_size, Scalar zero_scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_func", custom_func_execute);
}