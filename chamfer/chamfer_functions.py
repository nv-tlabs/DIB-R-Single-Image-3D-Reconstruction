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


def sample(verts, faces, num=10000, ret_choice=False):
    dist_uni = torch.distributions.Uniform(
        torch.tensor([0.0]).cuda(),
        torch.tensor([1.0]).cuda())
    # calculate area of each face
    x1, x2, x3 = torch.split(torch.index_select(verts, 0, faces[:, 0]) -
                             torch.index_select(verts, 0, faces[:, 1]),
                             1,
                             dim=1)
    y1, y2, y3 = torch.split(torch.index_select(verts, 0, faces[:, 1]) -
                             torch.index_select(verts, 0, faces[:, 2]),
                             1,
                             dim=1)
    a = (x2 * y3 - x3 * y2)**2
    b = (x3 * y1 - x1 * y3)**2
    c = (x1 * y2 - x2 * y1)**2
    Areas = torch.sqrt(a + b + c) / 2
    Areas = Areas / torch.sum(
        Areas)  # percentage of each face w.r.t. full surface area
    # define descrete distribution w.r.t. face area ratios caluclated
    cat_dist = torch.distributions.Categorical(Areas.view(-1))
    choices = cat_dist.sample_n(num)
    # from each face sample a point
    select_faces = faces[choices]
    xs = torch.index_select(verts, 0, select_faces[:, 0])
    ys = torch.index_select(verts, 0, select_faces[:, 1])
    zs = torch.index_select(verts, 0, select_faces[:, 2])
    u = torch.sqrt(dist_uni.sample_n(num))
    v = dist_uni.sample_n(num)
    points = (1 - u) * xs + (u * (1 - v)) * ys + u * v * zs
    if ret_choice:
        return points, choices
    else:
        return points


def normalize(vertices, maxratio):
    # import ipdb
    # ipdb.set_trace()
    max = vertices.max(0)
    min = vertices.min(0)
    # mean = np.mean(vertices, axis=0)
    vertices = vertices - (max + min) / 2
    pmax = np.abs(vertices).max()
    # maxratio = 0.45
    scale = maxratio / pmax
    vertices = vertices * scale
    return vertices
