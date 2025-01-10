/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include "convolution.h"

__global__ void _convolveKernel(float *input, float *kernel, float *output, int width, int height, int kernelSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * width + col;

    if (col < width && row < height) {
        float sum = 0.0f;
        int center = kernelSize / 2;

        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                int irow = row + i - center;
                int jcol = col + j - center;

                if ((irow >= 0 && irow < height) && (jcol >= 0 && jcol < width)) {

                    sum += input[irow * width + jcol] * kernel[i * kernelSize + j];
                }
            }
        }
        output[idx] = sum;
    }
}

void convolveKernel(float *input, float *kernel, float *output, int width, int height, int kernelSize, cudaStream_t stream) {

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocks((width+THREADS_PER_BLOCK_1)/threadsPerBlock.x, (height+THREADS_PER_BLOCK_1)/threadsPerBlock.y);

    _convolveKernel<<<blocks, threadsPerBlock, 0, stream>>>(input, kernel, output, width, height, kernelSize);
}
