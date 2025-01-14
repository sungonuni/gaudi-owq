# GEMV acceleration kernel for LLM inference on Intel Gaudi-2

This repository provides the TPC-C kernel code and sample test code with pytorch C++ extension of GEMV acceleration kernel for LLM inference, based on CUDA kernel of OWQ.

## Table Of Contents
* [TPC Kernels Overview](#tpc-kernels-overview)
* [Install Habanatools For Ubuntu](#install-habanatools-for-ubuntu)
* [Install GEMV acceleration kernel](#install-gemv-acceleration-kernel)

## TPC Kernels Overview

The Tensor Processor Core™ (**TPC**) is a fully programmable VLIW4 processor designed to execute non-linear deep learning operators. It is embedded in Habana’s Gaudi deep learning accelerator. Habana’s Gaudi SoC contains numerous TPC cores all operating in parallel, with each core running a single thread. The TPC is designed with very long instruction word (VLIW) architecture. It has a wide single instruction multiple data (SIMD) vector unit that support 2048-bit SIMD operations with data types such as float, bfloat16, INT16, INT32 and INT8. In each cycle, the TPC’s ALU (Arithmetic Logic Unit) can execute up to 64 floats/INT32 ops, or 128 INT16 ops, or 256 INT8 ops.
TPC is designed for workloads that do not map to Matrix Multiplication Engine (**MME**). Those workloads or operators can be implemented using TPC kernels. 

## Install Habanatools For Ubuntu
To retrieve the package please visit [Habana Vault](https://vault.habana.ai/artifactory/debian/jammy/pool/main/h/habanatools/habanatools_1.16.0-526_amd64.deb), click Artifact, find habanatools and download the latest release package for Ubuntu 22.04. You can find different packages for different OS you used. 
```  
  sudo dpkg -i ./habanatools_1.16.0-526_amd64.deb
```
- Once installed the following files will be added to your machine 
  
  |  |Location | Purpose  |
  |--|--------------------|-----------------------------|
  |1 | /usr/bin/tpc-clang | TPC-C compiler and assembler |
  |2 | /usr/bin/tpc-llvm-objdump | TPC dis-assembler|
  |3 | /usr/lib/habanatools/libtpcsim_shared.so | TPC simulator|
  |4 | /usr/lib/habanatools/libtpc_tests_core_ext.so | Test core library |  
  |5 | /usr/lib/habanatools/include/gc_interface.h | Glue code interface header |
  |6 | /usr/lib/habanatools/include/tpc_kernel_lib_interface.h | New TPC kernel GC2.0 interface header |
  |7 | /usr/lib/habanatools/include/tpc_test_core_api.h |Test core APIs |
  |8 | /usr/lib/habanatools/include/tpc_test_core_types.h | Test core type defines |  

## Install GEMV acceleration kernel

1. Build the custom TPC-C kernel codes at the top-level directory (/owq_tpc). We provide the integrated command by shell script. After the compile, you will get "libcustom_tpc_perf_lib.so" which includes the link of each custom kernels on build directory.
```  
bash build.sh
```  

2. (**IMPORTANT**) Register the path of custom TPC-C kernel file. The path of custom kernel should be added on environment variable GC_KERNEL_PATH. 
```  
export GC_KERNEL_PATH=/code/owq_tpc/build/src/libcustom_tpc_perf_lib.so:/usr/lib/habanalabs/libtpc_kernels.so:
```  

3. Build the pytorch C++ extension at the /owq_tpc/scripts/custom_op_test. After the compile, you will get "hpu_custom_func.cpython-310-x86_64-linux-gnu.so" which includes the link of pytorch custom extension on build directory.
```  
cd /owq_tpc/scripts/custom_op_test
python setup.py build
```  

4. Run the test code on /owq_tpc/scripts/custom_op_test.
```  
python test.py
```  