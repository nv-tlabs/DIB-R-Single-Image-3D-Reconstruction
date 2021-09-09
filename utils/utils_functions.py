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
import numpy as np

# render & camera
from kaolin.render.camera import perspective_camera
from kaolin.ops.mesh import index_vertices_by_faces, face_normals
from kaolin.render.mesh.rasterization import dibr_rasterization as dibr_rasterization_kaolin


############################################
# Help functions
############################################
def normalize_meshes_np(points_px3, mean_1x3=0, scale=1):
    r'''
    normalize the vertices in numpy
    '''
    p = points_px3
    pmax = np.max(p, axis=0, keepdims=True)
    pmin = np.min(p, axis=0, keepdims=True)
    pmiddle = (pmax + pmin) / 2
    p = p - pmiddle
    pmax = np.max(p, axis=0, keepdims=True)
    pmin = np.min(p, axis=0, keepdims=True)
    print('pmax {} pmin {}'.format(pmax[0], pmin[0]))

    pointnp_px3 = p * scale + mean_1x3
    pmax = np.max(pointnp_px3, axis=0, keepdims=True)
    pmin = np.min(pointnp_px3, axis=0, keepdims=True)
    print('pmax {} pmin {}'.format(pmax[0], pmin[0]))
    return pointnp_px3


def normalize_meshes(mesh_bxpx3):
    r'''
    normalize the vertices in pytorch
    '''
    mesh_max = torch.max(mesh_bxpx3, dim=1, keepdim=True)[0]
    mesh_min = torch.min(mesh_bxpx3, dim=1, keepdim=True)[0]
    mesh_middle = (mesh_min + mesh_max) / 2
    mesh_bxpx3 = mesh_bxpx3 - mesh_middle

    bs = mesh_bxpx3.shape[0]
    mesh_biggest = torch.max(mesh_bxpx3.view(bs, -1), dim=1)[0]
    mesh_bxpx3 = mesh_bxpx3 / mesh_biggest.view(bs, 1, 1)

    # 0.45 for dibr
    # 0.5 for NMR
    return mesh_bxpx3 * 0.5


############################################
# render functions
############################################
def render_vertex_colors(vertices_camera, faces, vertex_colors, camera_proj,
                         height, width):
    r'''
    render the vertices and colors to the images 
    '''

    # face_vertex_colors
    attributes = vertex_colors
    face_attributes_idx = faces
    face_attributes = index_vertices_by_faces(attributes, face_attributes_idx)

    # normals
    vertices_image = perspective_camera(vertices_camera, camera_proj)
    face_vertices_camera = index_vertices_by_faces(vertices_camera, faces)
    face_vertices_z = face_vertices_camera[:, :, :, 2]

    face_camera_normals = face_normals(face_vertices_camera, unit=True)
    face_camera_normals_z = face_camera_normals[:, :, 2:3]
    face_vertices_image = index_vertices_by_faces(vertices_image, faces)

    imfeat, improb, imfaceidx = dibr_rasterization_kaolin(
        height, width, face_vertices_z, face_vertices_image, face_attributes,
        face_camera_normals_z)
    improb = improb.unsqueeze(3)

    image = imfeat

    return image, improb, face_camera_normals


############################################
# loss functions
############################################
def calculate_iou_loss(gt_mask_bx1xhxw,
                       pred_mask_bx1xhxw,
                       lossname='iou',
                       eps=1e-10):
    r'''
    calculate mask loss, generally iou is better than l1
    '''
    if lossname == 'iou':
        bs = pred_mask_bx1xhxw.shape[0]
        silhouette_mul = pred_mask_bx1xhxw * gt_mask_bx1xhxw
        silhouette_add = pred_mask_bx1xhxw + gt_mask_bx1xhxw
        silhouette_mul = silhouette_mul.view(bs, -1)
        silhouette_add = silhouette_add.view(bs, -1)
        iouup = torch.sum(silhouette_mul, dim=1)
        ioudown = torch.sum(silhouette_add - silhouette_mul, dim=1)
        iou = iouup / (ioudown + eps)
        silhouette_loss = 1.0 - torch.mean(iou)
    elif lossname == 'l1':
        silhouette_loss = (pred_mask_bx1xhxw - gt_mask_bx1xhxw).abs().mean()

    return silhouette_loss
