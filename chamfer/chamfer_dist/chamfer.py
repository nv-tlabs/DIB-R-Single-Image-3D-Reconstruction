# Copyright (c) 2020,21 NVIDIA CORPORATION & AFFILIATES.. All rights reserved.
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

import torch

from torch.utils.cpp_extension import load

cd = load(name="chamfer",
          sources=["chamfer_dist/chamfer.cpp", "chamfer_dist/chamfer.cu"])


class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n).float().to(xyz1.device)
        dist2 = torch.zeros(batchsize, m).float().to(xyz1.device)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor).to(xyz1.device)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor).to(xyz1.device)

        cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)
        return idx1, idx2, dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        ints = ctx.saved_tensors
        gradxyz1 = torch.zeros(ints.size())
        return gradxyz1, gradxyz1


class Chamfer(torch.nn.Module):
    def forward(self, points1, points2):
        return ChamferFunction.apply(points1, points2)
