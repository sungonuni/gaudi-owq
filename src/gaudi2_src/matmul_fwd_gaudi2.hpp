#define _MATRIX_MUL_FWD_F32_HPP

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class MatrixMulFwdF32Gaudi2
{
    public:
        MatrixMulFwdF32Gaudi2() {}
        virtual ~MatrixMulFwdF32Gaudi2() {}

        virtual tpc_lib_api::GlueCodeReturn
        GetGcDefinitions(tpc_lib_api::HabanaKernelParams*      in_defs,
                     tpc_lib_api::HabanaKernelInstantiation* out_defs);

        virtual tpc_lib_api::GlueCodeReturn GetKernelName(
                char kernelName [tpc_lib_api::MAX_NODE_NAME]);                            

    private:
        MatrixMulFwdF32Gaudi2(const MatrixMulFwdF32Gaudi2& other) = delete;
        MatrixMulFwdF32Gaudi2& operator=(const MatrixMulFwdF32Gaudi2& other) = delete;
};
