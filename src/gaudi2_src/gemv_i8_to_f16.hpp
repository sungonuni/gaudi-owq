#ifndef _GEMV_I8_GAUDI2_HPP
#define _GEMV_I8_GAUDI2_HPP

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class GemvDequantInt8Gaudi2
{
public:
    GemvDequantInt8Gaudi2() {}
    virtual ~GemvDequantInt8Gaudi2() {}

    virtual tpc_lib_api::GlueCodeReturn GetGcDefinitions(
            tpc_lib_api::HabanaKernelParams* params,
            tpc_lib_api::HabanaKernelInstantiation* kernel);

    virtual tpc_lib_api::GlueCodeReturn GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME]);

private:
    GemvDequantInt8Gaudi2(const GemvDequantInt8Gaudi2& other) = delete;
    GemvDequantInt8Gaudi2& operator=(const GemvDequantInt8Gaudi2& other) = delete;
};


#endif //_RELU_ALL_GAUDI2_HPP