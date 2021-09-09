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

import sys

sys.path.append('..')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# utils
from utils.utils_mesh import loadobj, face2edge, face2edge2, face2pneimtx, edge2face, mtx2tfsparse
from utils.utils_functions import normalize_meshes_np
# we use the same network arthichetecture as DVR
from models.models import DVR

from kaolin.render.camera import generate_perspective_projection
from kaolin.visualize import Timelapse


#############################################################################
# io operation
def create_svfolder(args):
    ####################
    # Make directories #
    ####################

    if not os.path.isdir(args.svfolder):
        os.mkdir(args.svfolder)

    # main folder
    i = 0
    svfolder = os.path.join('%s' % args.svfolder, 'dibr-%d' % i)

    while os.path.exists(svfolder):
        i += 1
        svfolder = os.path.join('%s' % args.svfolder, 'dibr-%d' % i)
    os.mkdir(svfolder)

    # Directory for samples
    if not os.path.exists(os.path.join(svfolder, 'samples')):
        os.mkdir(os.path.join(svfolder, 'samples'))

    # Directory for checkpoints
    if not os.path.exists(os.path.join(svfolder, 'checkpoints')):
        os.mkdir(os.path.join(svfolder, 'checkpoints'))

    return svfolder


#############################################################################
# model operation
def prepare_template(file_name='sphere_3.obj'):
    '''
    load a template_vertices mesh
    template generated by meshlab with subdivision level of 3
    return the points, mesh, edge connection, adjacent matrix,
    '''

    # load mesh
    pointnp_px3, facenp_fx3 = loadobj(file_name)
    # edge connection indexed by vertices
    edgelist = face2edge(facenp_fx3)
    # edge connection indexed by faces
    edgelist = edge2face(facenp_fx3, edgelist)
    # vertice adjacent matrix, dense
    pneimtx = face2pneimtx(facenp_fx3)

    # normalize the mesh
    pointnp_px3 = normalize_meshes_np(pointnp_px3, scale=0.35 / 0.45 * 0.5)

    # convert to pytorch data
    point_px3 = torch.from_numpy(pointnp_px3)
    template_face = torch.from_numpy(facenp_fx3)
    edgelist = torch.from_numpy(edgelist)
    pointadj_mtx = mtx2tfsparse(pneimtx)

    return point_px3, template_face, edgelist, pointadj_mtx


################################################
# template_vertices mesh & models
################################################
def create_model(args, svfolder):

    template_vertices, template_face, edgelist, pointadj_mtx = prepare_template(
    )
    # NMR use 30 degree fovy angle
    camproj_mtx = generate_perspective_projection(fovyangle=30 / 180.0 * np.pi,
                                                  ratio=1.0)

    # Create models
    g_model_im2mesh = DVR(template=template_vertices)

    return g_model_im2mesh, \
        template_vertices, template_face, edgelist, pointadj_mtx, camproj_mtx


def load_model(args, svfolder, g_model_im2mesh):
    # Try loading the latest existing checkpoints based on iter_num
    if args.iterbe != -1:
        # Checkpoint dirs
        g_model_dir = os.path.join(svfolder, 'checkpoints',
                                   'g_model_' + str(args.iterbe) + '.pth')
        g_model_im2mesh.load_state_dict(torch.load(g_model_dir))
        print('Loaded the latest checkpoints from {}-th iteration.'.format(
            args.iterbe))
        print(
            'NOTE: Set the `BEGIN_EPOCH` in accordance to saved checkpoints.')
    else:
        print("Train from scratch.")

    return


################################################
# Define device, neural nets, optimizers, etc. #
################################################
def create_optimizer(args, svfolder, g_model_im2mesh):

    # Optimizers
    g_params = list(g_model_im2mesh.parameters())
    g_optim = optim.Adam(g_params, args.lr, betas=(0.9, 0.999))
    lrschedule = optim.lr_scheduler.StepLR(g_optim,
                                           step_size=args.decaystep,
                                           gamma=args.lrdecay,
                                           last_epoch=-1)

    writer = SummaryWriter(os.path.join(svfolder, 'logs_train'))
    timelapse = Timelapse(os.path.join(svfolder, 'visualize_train'))

    return g_optim, lrschedule, writer, timelapse

def load_device(args, device, g_model_im2mesh, \
                     template_face, edgelist, pointadj_mtx, camproj_mtx):

    g_model_im2mesh.to(device)

    template_face = template_face.to(device)
    edgelist = edgelist.to(device)
    pointadj_mtx = pointadj_mtx.to(device)
    camproj_mtx = camproj_mtx.to(device)

    return template_face, edgelist, pointadj_mtx, camproj_mtx
