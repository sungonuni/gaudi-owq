#define FLOAT32
#include "kernel_config.h"
#pragma tpc_printf (enable)

void main(tensor input, tensor output, float step_size, float zero_point)
{
  const int depth = 0;
  const int width = 1;
  const int height = 2;
  const int batch = 3;
  const int fifthDim = 4;

  const int5 indexSpaceStart = get_index_space_offset();
  const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

  int5 ifmCoords = {0, 0, 0, 0, 0};

  // DEPTH
  const int depthStep = VECTOR_SIZE;
  const int depthStart = indexSpaceStart[depth] * depthStep;
  const int depthEnd = indexSpaceEnd[depth] * depthStep;

  // WIDTH
  const int widthStep = 4;
  const int widthStart = indexSpaceStart[width] * widthStep;
  const int widthEnd = indexSpaceEnd[width] * widthStep;

  // HEIGHT
  const int heightStep = 1;
  const int heightStart = indexSpaceStart[height];
  const int heightEnd = indexSpaceEnd[height];

  // BATCH
  const int batchStep = 1;
  const int batchStart = indexSpaceStart[batch];
  const int batchEnd = indexSpaceEnd[batch];

  // fifthDim
  const int fifthDimStep = 1;
  const int fifthDimStart = indexSpaceStart[fifthDim];
  const int fifthDimEnd = indexSpaceEnd[fifthDim];

  VECTOR x00;
  VECTOR o00;

#pragma loop_taken
  for (int d = depthStart; d < depthEnd; d += depthStep) {
    ifmCoords[depth] = d;

#pragma loop_taken
    for (int f = fifthDimStart; f < fifthDimEnd; f += fifthDimStep) {
      ifmCoords[fifthDim] = f;

#pragma loop_taken
      for (int b = batchStart; b < batchEnd; b += batchStep) {
        ifmCoords[batch] = b;

#pragma loop_taken
        for (int h = heightStart; h < heightEnd; h += heightStep) {
          ifmCoords[height] = h;

#pragma loop_taken
#pragma unroll 4
          for (int w = widthStart; w < widthEnd; w += 1) {
            ifmCoords[width] = w;

            x00 = v_ld_tnsr_i(ifmCoords, input);
            // printf("step_size value is %f\n", (SCALAR)step_size);
            // printf("zero_point value is %f\n", (SCALAR)zero_point);
            o00 = v_mul_v_s(x00, (SCALAR)step_size);
            o00 = v_f32_add_b(o00, (SCALAR)zero_point);

            // store
            st_tnsr_i_v(ifmCoords, output, o00);
          }
        }
      }
    }
  }
}