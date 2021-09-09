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

# utils functions

import torch

import sys

sys.path.append('..')
from utils.utils_functions import normalize_meshes
from kaolin.render.camera import rotate_translate_points


def pred_meshes(args, g_model_im2mesh, gtims):
    r'''
    Predicting meshes from images
    '''

    meshes_pred = []
    meshmovs_pred = []
    meshcolors_pred = []

    # generate j-th mesh
    for j in range(args.viewnum):
        # predict mesh, movement from rgb images
        mesh_bxpx3, meshmov_bxpx3, meshcolor_bxpx3 = g_model_im2mesh(
            gtims[j][:, :3, :, :])

        mesh_bxpx3 = normalize_meshes(mesh_bxpx3)

        meshes_pred.append(mesh_bxpx3)
        meshmovs_pred.append(meshmov_bxpx3)
        meshcolors_pred.append(meshcolor_bxpx3)

    return meshes_pred, meshmovs_pred, meshcolors_pred


def rotate_meshes(args, meshes_pred, meshcolors_pred, gtims, camrots,
                  camposes):
    r'''
    Rotating the meshes to another views
    '''

    meshes_render = []
    meshcolors_render = []
    gtims_render = []

    for j in range(args.viewnum):
        # use j-th mesh
        meshworld_bxpx3 = meshes_pred[j]
        meshcolor_bxpx3 = meshcolors_pred[j]

        for k in range(args.viewnum):

            # render under k-th camera
            camrot_bx3x3, campos_bx3 = camrots[k], camposes[k]

            meshcamera_bxpx3 = rotate_translate_points(meshworld_bxpx3,
                                                       camrot_bx3x3,
                                                       campos_bx3)

            # GT image
            gtim_bx4xhxw = gtims[k]

            meshes_render.append(meshcamera_bxpx3)
            meshcolors_render.append(meshcolor_bxpx3)
            gtims_render.append(gtim_bx4xhxw)

    meshes_render = torch.cat(meshes_render)
    meshcolors_render = torch.cat(meshcolors_render)
    gtims_render = torch.cat(gtims_render)

    return meshes_render, meshcolors_render, gtims_render
