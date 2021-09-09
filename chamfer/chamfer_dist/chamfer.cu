/*
 * # Copyright (c) 2020,21 NVIDIA CORPORATION & AFFILIATES.. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 */

 
#include <ATen/ATen.h>
#include <iostream>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ChamferKernel(
	const float* xyz1,
    const float* xyz2,
    float* dist,
    int* idx, int batch_size, int n, int m)
{
    // bidx * height + heiidx
	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
	int n_idx = presentthread % n;
	int b_idx = (presentthread - n_idx) / n;

	if (b_idx >= batch_size || n_idx >= n) {
		return;
	}
	int min_idx = 0;
	float min_dist = 10000.0;
	float cur_x = xyz1[b_idx * n * 3 + n_idx * 3];
	float cur_y = xyz1[b_idx * n * 3 + n_idx * 3 + 1];
	float cur_z = xyz1[b_idx * n * 3 + n_idx * 3 + 2];
	float next_x, next_y, next_z;
	float diff_x, diff_y, diff_z;
	float tmp_dist;
    for (int i = 0; i < m; i++){
        next_x = xyz2[b_idx * m * 3 + i * 3];
        next_y = xyz2[b_idx * m * 3 + i * 3 + 1];
        next_z = xyz2[b_idx * m * 3 + i * 3 + 2];

        diff_x = cur_x - next_x;
        diff_y = cur_y - next_y;
        diff_z = cur_z - next_z;

        tmp_dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        tmp_dist = sqrt(tmp_dist);
        if (tmp_dist < min_dist){
            min_dist = tmp_dist;
            min_idx = i;
        }
    }
    dist[b_idx * n + n_idx] = min_dist;
    idx[b_idx * n + n_idx] = min_idx;
}

void ChamferKernelLauncher(
    const float* xyz1,
    const float* xyz2,
    float* dist1,
    float* dist2,
    int* idx1,
    int* idx2,
    int batch_size, int n, int m){

    const int threadnum = 1024;
	const int totalthread = batch_size * n;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	ChamferKernel<<<blocks, threads>>>(xyz1, xyz2, dist1, idx1, batch_size, n, m);
	const int totalthread2 = batch_size * m;
	const int blocknum2 = totalthread2 / threadnum + 1;

    const dim3 threads2(threadnum, 1, 1);
	const dim3 blocks2(blocknum2, 1, 1);
	ChamferKernel<<<blocks2, threads2>>>(xyz2, xyz1, dist2, idx2, batch_size, m, n);

	cudaError_t err = cudaGetLastError();
}

