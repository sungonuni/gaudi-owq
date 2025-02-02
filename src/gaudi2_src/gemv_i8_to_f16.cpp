#include <cstring>
#include "gemv_i8_to_f16.hpp"

extern unsigned char _binary___gemv_i8_to_f16_o_start; // should be same with the name of compiled .c (.o) file
extern unsigned char _binary___gemv_i8_to_f16_o_end; // should be same with the name of compiled .c (.o) file

tpc_lib_api::GlueCodeReturn GemvDequantInt8Gaudi2::GetKernelName(
        char kernelName [tpc_lib_api::MAX_NODE_NAME])
{
    strcpy(kernelName,"custom_gemv_deq_int8_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn GemvDequantInt8Gaudi2::GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* in_defs,
        tpc_lib_api::HabanaKernelInstantiation* out_defs)
{
	const int c_unrollCount = 4;
    tpc_lib_api::GlueCodeReturn retVal;
    gemvParam* def = static_cast<gemvParam*>(in_defs->nodeParams.nodeParams);
    /*************************************************************************************
    *   Stage I - validate input
    **************************************************************************************/
    //validate correct amount of input tensors
    if (in_defs->inputTensorNr != 1)
    {
        in_defs->inputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }

    //validate correct amount of output tensors
    if (in_defs->outputTensorNr != 1)
    {
        in_defs->outputTensorNr  = 1;
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }

    // validate input and output data type
    if (in_defs->inputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32 ||
        in_defs->outputTensors[0].geometry.dataType != tpc_lib_api::DATA_F32)
    {
        in_defs->inputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        in_defs->outputTensors[0].geometry.dataType = tpc_lib_api::DATA_F32;
        return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    // Tensor 0 should be input feature map.
    // The semantics of the input tensors and their order is a convention
    // between TPC kernel writer and the write of the layer at the
    // framework level.
    uint64_t outputSizes[gcapi::MAX_TENSOR_DIM] = {0};

    memcpy(outputSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));

    // verify that output feature map dimension are correct
    if (memcmp(in_defs->outputTensors[0].geometry.maxSizes, outputSizes,
               in_defs->outputTensors[0].geometry.dims * sizeof(uint64_t) ) != 0)
    {
        memcpy(in_defs->outputTensors[0].geometry.maxSizes, in_defs->inputTensors[0].geometry.maxSizes, sizeof(outputSizes));
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }

    /*************************************************************************************
    *    Stage II -  Define index space geometry. In this example the index space matches
    *    the dimensions of the output tensor, up to dim 0.
    **************************************************************************************/
    int elementsInVec = 64;

    //round up to elementsInVec and divide by elementsInVec.
    unsigned depthIndex = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
    out_defs->indexSpaceRank = 5;
    out_defs->indexSpaceGeometry[0] = depthIndex;
	//reduce index space due to unroll.
    out_defs->indexSpaceGeometry[1] = (outputSizes[1] +(c_unrollCount-1)) / c_unrollCount; 
    out_defs->indexSpaceGeometry[2] = outputSizes[2];
    out_defs->indexSpaceGeometry[3] = outputSizes[3];
    out_defs->indexSpaceGeometry[4] = outputSizes[4];

    /*************************************************************************************
    *    Stage III -  Define index space mapping
    **************************************************************************************/
    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (IFM) dim 0
    out_defs->inputTensorAccessPattern[0].allRequired = true;
    out_defs->inputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->inputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;

	out_defs->inputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    out_defs->inputTensorAccessPattern[0].mapping[1].a        = c_unrollCount;
    out_defs->inputTensorAccessPattern[0].mapping[1].start_b  = 0;
    out_defs->inputTensorAccessPattern[0].mapping[1].end_b    = c_unrollCount - 1;
	
    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (IFM) dim 1-4
    for (unsigned int dims = 2; dims < out_defs->indexSpaceRank; dims++)
    {
        out_defs->inputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        out_defs->inputTensorAccessPattern[0].mapping[dims].a        = 1;
        out_defs->inputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        out_defs->inputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }

    //out_defs->inputTensorAccessPattern[1].allRequired = true;
    out_defs->inputTensorAccessPattern[1].mapping[0].indexSpaceDim      = 0;
    out_defs->inputTensorAccessPattern[1].mapping[0].a        = elementsInVec;
    out_defs->inputTensorAccessPattern[1].mapping[0].start_b  = 0;
    out_defs->inputTensorAccessPattern[1].mapping[0].end_b    = elementsInVec - 1;

	out_defs->inputTensorAccessPattern[1].mapping[1].indexSpaceDim      = 1;
    out_defs->inputTensorAccessPattern[1].mapping[1].a        = c_unrollCount;
    out_defs->inputTensorAccessPattern[1].mapping[1].start_b  = 0;
    out_defs->inputTensorAccessPattern[1].mapping[1].end_b    = c_unrollCount - 1;
	
    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (IFM) dim 1-4
    for (unsigned int dims = 2; dims < out_defs->indexSpaceRank; dims++)
    {
        out_defs->inputTensorAccessPattern[1].mapping[dims].indexSpaceDim      = dims;
        out_defs->inputTensorAccessPattern[1].mapping[dims].a        = 1;
        out_defs->inputTensorAccessPattern[1].mapping[dims].start_b  = 0;
        out_defs->inputTensorAccessPattern[1].mapping[dims].end_b    = 1 - 1;
    }

    // f_start f(i) = elementsInVec*i + 0;
    // f_end   f(i) = elementsInVec*i + (elementsInVec - 1);
    // Resource 0 (OFM) dim 0
    out_defs->outputTensorAccessPattern[0].mapping[0].indexSpaceDim      = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].a        = elementsInVec;
    out_defs->outputTensorAccessPattern[0].mapping[0].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[0].end_b    = elementsInVec - 1;
	
	out_defs->outputTensorAccessPattern[0].mapping[1].indexSpaceDim      = 1;
    out_defs->outputTensorAccessPattern[0].mapping[1].a        = c_unrollCount;
    out_defs->outputTensorAccessPattern[0].mapping[1].start_b  = 0;
    out_defs->outputTensorAccessPattern[0].mapping[1].end_b    = c_unrollCount - 1;

    // f_start f(i) = 1*i + 0;
    // f_end   f(i) = 1*i + 0;
    // Resource 0 (OFM) dim 1-4
    for (unsigned int dims = 2; dims < out_defs->indexSpaceRank; dims++)
    {
        out_defs->outputTensorAccessPattern[0].mapping[dims].indexSpaceDim      = dims;
        out_defs->outputTensorAccessPattern[0].mapping[dims].a        = 1;
        out_defs->outputTensorAccessPattern[0].mapping[dims].start_b  = 0;
        out_defs->outputTensorAccessPattern[0].mapping[dims].end_b    = 1 - 1;
    }


    /*************************************************************************************
    *    Stage IV -  define scalar parameters
    **************************************************************************************/
    out_defs->kernel.paramsNr = sizeof(*def)/ sizeof(int);
    memcpy(&( out_defs->kernel.scalarParams[0]),def, sizeof(*def));

    /*************************************************************************************
    *    Stage V -  Load ISA into the descriptor.
    **************************************************************************************/
    unsigned IsaSize = (&_binary___gemv_i8_to_f16_o_end - &_binary___gemv_i8_to_f16_o_start);
    unsigned char *binary_kernel =  &_binary___gemv_i8_to_f16_o_start;
    unsigned givenBinarySize = out_defs->kernel.elfSize;
    out_defs->kernel.elfSize = IsaSize;
    if (givenBinarySize >= IsaSize)
    {
        memcpy (out_defs->kernel.kernelElf, binary_kernel, IsaSize);
    }
    else
    {
        retVal = tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
        return retVal;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}