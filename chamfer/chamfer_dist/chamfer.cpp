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


#include <torch/torch.h>
#include <iostream>
using namespace std;
// CUDA forward declarations

void ChamferKernelLauncher(
    const float* xyz1,
    const float* xyz2,
    float* dist1,
    float* dist2,
    int* idx1,
    int* idx2,
    int b, int n, int m);

void chamfer_forward_cuda(
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    at::Tensor dist1,
    at::Tensor dist2,
    at::Tensor idx1,
    at::Tensor idx2)
{
    int batch_size = xyz1.size(0);
    int n = xyz1.size(1);
    int m = xyz2.size(1);
    ChamferKernelLauncher(xyz1.data<float>(), xyz2.data<float>(),
                                            dist1.data<float>(), dist2.data<float>(),
                                            idx1.data<int>(), idx2.data<int>(), batch_size, n, m);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &chamfer_forward_cuda, "Chamfer forward (CUDA)");
}
