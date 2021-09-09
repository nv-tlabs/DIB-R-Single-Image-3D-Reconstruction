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

from __future__ import print_function
from __future__ import division

# general
import os
import cv2
import numpy as np

# pytorch
import torch
import torchvision.utils as vutils

# utils functions
import sys

sys.path.append('..')
from train_prepare import create_svfolder, create_model, create_optimizer, load_model,\
    load_device
from dataloader.dataloader_utils import prepare_dataloader, parse_dataloader
from train_functions import pred_meshes, rotate_meshes
from utils.utils_functions import render_vertex_colors, calculate_iou_loss
from utils.utils_mesh import savemeshcolor

############################################
# Make experiments reproducible
############################################
_ = torch.manual_seed(1234569527)
np.random.seed(1234569527)

# image size
IMG_DIM = 224
# how many iterations to print loss
ITERS_PER_LOG = 1
# how many iterations to save sampled images
ITERS_PER_SAMPLE = 1


def test_epoch(args, g_model_im2mesh, \
                faces, camera_intrinstic, pointadj_mtx, edge_list, \
                dataloader, svfolder, device, iter_num=-1):
    '''
    Args:
        args: training parameters
        g_model_im2mesh: network model
        g_optim: network optimizer, adam for default
        faces: template_vertices mesh faces, of shape (num_vertices, 3)
        camera_intrinstic: perspective camera proejction matrix, of shape (3, 1)
        pointadj_mtx: template_vertices mesh adjacent matrix, used to calculate laplacian loss
        edge_list: template_vertices mesh edge connection matrix, indexed by face, used to calculate flattern loss
        dataloader: shapenet dataloader
        writer: tensorboardX logger
        svfolder: save folder to store log
        device: training devices, dibr only supports cuda
        iter_num: training iteration number for logger
    '''

    for i, data_batch, in enumerate(dataloader):
        # print(i)
        if data_batch is None:
            continue

        iter_num += 1
        # print(iter_num)

        gtims, camrots, camposes = parse_dataloader(data_batch, args.viewnum,
                                                    device)

        ########################################3
        # pred meshes
        meshes_pred, meshmovs_pred, meshcolors_pred = pred_meshes(
            args, g_model_im2mesh, gtims)

        # rotate meshes in different views
        meshes_render, meshcolors_render, gtims_render = \
        rotate_meshes(args, meshes_pred, meshcolors_pred, gtims, camrots, camposes)

        pred_ims, pred_masks, meshnormals = \
        render_vertex_colors(meshes_render, faces, meshcolors_render, camera_intrinstic, IMG_DIM, IMG_DIM)

        ##############################################################
        # Compute loss
        gt_ims = gtims_render[:, :3, :, :]
        gt_masks = gtims_render[:, 3:4, :, :]

        pred_ims = pred_ims.permute(0, 3, 1, 2)
        pred_masks = pred_masks.permute(0, 3, 1, 2)

        #############################################################################
        # 1 mask loss
        silhouette_loss = calculate_iou_loss(gt_masks, pred_masks)

        # 2 color loss
        color_loss = torch.mean(torch.abs(pred_ims - gt_ims) * gt_masks)

        # 3 laplacian
        meshmovs_pred = torch.cat(meshmovs_pred)
        neighbourmovs = torch.stack(
            [torch.matmul(pointadj_mtx, da) for da in meshmovs_pred], dim=0)
        laplacian_loss = torch.mean(
            (neighbourmovs - meshmovs_pred)**2) * meshmovs_pred.shape[1] * 3

        # 4 flat loss for all the connected faces
        meshnormal_e1 = meshnormals[:, edge_list[:, 0], :]
        meshnormal_e2 = meshnormals[:, edge_list[:, 1], :]
        facecos = torch.sum(meshnormal_e1 * meshnormal_e2, dim=2)
        flattern_loss = torch.mean((facecos - 1)**2) * edge_list.shape[0]

        g_loss = 3.0 * color_loss + 1.0 * silhouette_loss + 0.01 * laplacian_loss + 0.001 * flattern_loss

        ##################
        # Log statistics #
        ##################
        # Print statistics
        if iter_num % ITERS_PER_LOG == ITERS_PER_LOG - 1:
            print('epo: {}, iter: {}, g_loss: {}, color_loss: {}, iou_loss: {}, lap_loss: {}, flat_loss: {}'\
                  .format(epo, iter_num, g_loss, color_loss, silhouette_loss, laplacian_loss, flattern_loss))

        # Save image grids of fake and real images
        if iter_num % ITERS_PER_SAMPLE == ITERS_PER_SAMPLE - 1:

            pred_masks = pred_masks.repeat(1, 3, 1, 1)
            gt_masks = gt_masks.repeat(1, 3, 1, 1)
            re = torch.cat((gt_ims, gt_masks, pred_ims, pred_masks), dim=3)

            real_samples_dir = os.path.join(
                svfolder, 'real_{:0>7d}.png'.format(iter_num))
            vutils.save_image(re, real_samples_dir, normalize=False)

            if True:
                ###########################################
                # save meshes
                md5names = data_batch['md5']
                cats = data_batch['cate']
                viewnums = data_batch['view0']['viewidx']

                # prediction
                pointsnp_bxpx3 = meshes_pred[0].detach().cpu().numpy()
                colorsnp_bxpx3 = meshcolors_pred[0].detach().cpu().numpy()
                facenp_fx3 = faces.detach().cpu().numpy()
                renp_bx3xhxw = re.detach().cpu().numpy()

                n_batch = pointsnp_bxpx3.shape[0]
                for i_batch in range(n_batch):
                    viewid = viewnums[i_batch]
                    # assert viewid == 0

                    catname = cats[i_batch]
                    obj_name = md5names[i_batch].split('/')[-1]
                    print('class %s md5 %s view %d' %
                          (catname, obj_name, viewid))

                    colorsnp_px3 = colorsnp_bxpx3[i_batch, :, ::-1]
                    tet_p = pointsnp_bxpx3[i_batch]
                    fname = os.path.join(svfolder, catname,
                                         '%s_%d.obj' % (obj_name, viewid))
                    savemeshcolor(tet_p, facenp_fx3, fname, colorsnp_px3)

                    imnp = renp_bx3xhxw[i_batch]
                    imnp = np.transpose(imnp, [1, 2, 0])
                    fname = os.path.join(svfolder, catname,
                                         '%s_%d.png' % (obj_name, viewid))
                    cv2.imwrite(fname, imnp * 255)

    return iter_num


if __name__ == '__main__':

    # usage
    # python eval.py --datafolder "YOU RDATASET" --svfolder "YOUR SAVING FOLDER" --iterbe "YOUR CHECKPOINT" --viewnum 1
    from config import get_args
    args = get_args()

    ####################
    # Make directories #
    ####################
    # should be the folder contains checkpoints, samples and logs
    svfolder = args.svfolder

    g_model_im2mesh, \
    template_vertices, template_face, edgelist, pointadj_mtx, camproj_mtx \
    = create_model(args, svfolder)

    # Try loading the latest existing checkpoints based on iter_num
    load_model(args, svfolder, g_model_im2mesh)

    # Automatic GPU/CPU device placement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    template_face, edgelist, pointadj_mtx, camproj_mtx = \
    load_device(args, device, g_model_im2mesh, template_face, edgelist, pointadj_mtx, camproj_mtx)

    dataloader, cls = prepare_dataloader(args, mode='test')

    ############
    # Testing #
    ############

    print('Begin testing!')
    imsvfolder = os.path.join(svfolder, 'test-%d' % args.iterbe)
    if not os.path.isdir(imsvfolder):
        os.mkdir(imsvfolder)

    # svfolder for each calss
    for cl in cls:
        objsvdir = '%s/%s' % (imsvfolder, cl)
        if not os.path.isdir(objsvdir):
            os.mkdir(objsvdir)

    g_model_im2mesh.eval()

    for epo in range(1):
        test_epoch(args, g_model_im2mesh, \
            template_face, camproj_mtx, pointadj_mtx, edgelist, \
            dataloader, imsvfolder, device, iter_num=0)
