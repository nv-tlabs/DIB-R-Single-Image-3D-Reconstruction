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

import os
import glob
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description='check chamfer-dist and F-score')

    parser.add_argument('--folder',
                        type=str,
                        default='YOUR MODEL FOLDER',
                        help='prediction folder')
    parser.add_argument('--ext',
                        type=str,
                        default='obj',
                        help='mesh files, do not has .!!!')

    parser.add_argument('--norm',
                        type=float,
                        default=-1,
                        help='do we normalize the mesh')
    parser.add_argument('--N',
                        type=int,
                        default=2500,
                        help='number of sampled points, less than 10000')

    parser.add_argument('--gt_folder',
                        type=str,
                        default='YOUR DATASET FOLDER',
                        help='gt folder')

    parser.add_argument('--f',
                        type=str,
                        default='',
                        help='no use, make notebook happy')

    # args = parser.parse_args(args=['--f', '0'])
    args = parser.parse_args()
    return args


import numpy as np
from tqdm import tqdm
from chamfer_functions import Chamfer, sample

torch.random.manual_seed(12345)
np.random.seed(123456)
import random

random.seed(1234567)
from collections import defaultdict
import pickle


def loadobj(meshfile):

    v = []
    f = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4 and len(data) != 7:
            continue
        if data[0] == 'v':
            v.append([float(d) for d in data[1:4]])
        if data[0] == 'f':
            data = [da.split('/')[0] for da in data]
            f.append([int(d) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    return pointnp_px3, facenp_fx3


def main():
    args = get_args()
    folder = args.folder
    ext = args.ext
    norm = args.norm
    N = args.N
    print('==> get all predictions')
    print('folder :%s' % folder)
    meshfiles = sorted(glob.glob('%s/*/*.%s' % (folder, ext)))
    print('Length mesh files: ', len(meshfiles))
    gt_folder = args.gt_folder
    print('gt folder :%s' % gt_folder)

    print('==> starting ')
    chamfer = Chamfer()
    cates = '02691156,02828884,02933112,02958343,03001627,03211117,03636649,03691459,04090263,04256520,04379243,04401088,04530566'
    cates = cates.split(',')
    dist_cate = defaultdict(list)
    F_score = defaultdict(list)

    random.seed(1)
    random.shuffle(meshfiles)
    toa = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    for i, fl in enumerate(tqdm(meshfiles)):
        tmp_name = fl
        try:
            vertices, faces = loadobj(tmp_name)
        except:
            print('Error in read shape')
            print(tmp_name)
            continue

        if norm > 0:
            vertices = normalize(vertices, norm)

        vertices = torch.from_numpy(vertices).float().cuda()
        faces = torch.from_numpy(faces).cuda()
        try:
            sample_p = sample(vertices, faces, num=N)
            sample_p = sample_p.unsqueeze(0)
        except:
            print('Error in sample:')
            print(tmp_name)
            continue

        #####################################
        names = tmp_name.split('/')
        md5 = names[-1].split('.')[0].split('_')[0]
        cat = names[-2]
        # dvr dataset
        gt_name = os.path.join(gt_folder, cat, md5, 'pointcloud.npz')
        # normalized, -0.5, 0.5
        try:
            gt_vertices = np.load(gt_name)['points']
            gt_vertices = gt_vertices[np.random.permutation(
                gt_vertices.shape[0])[:N]]
        except:
            print('Error in GT:')
            print(tmp_name)
            continue

        sample_gt = torch.from_numpy(gt_vertices).float().cuda()
        sample_gt = sample_gt.unsqueeze(0)

        ################################################
        _, _, dist1, dist2 = chamfer(sample_p, sample_gt)
        cf = (dist1.mean() + dist2.mean()) / 2
        f_score_list = []
        for t in toa:
            fp = (dist1 > t).float().sum()
            tp = (dist1 <= t).float().sum()
            precision = tp / (tp + fp)
            tp = (dist2 <= t).float().sum()
            fn = (dist2 > t).float().sum()
            recall = tp / (tp + fn)
            f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            f_score_list.append(f_score.item())
        F_score[cat].append(f_score_list)
        dist_cate[cat].append(cf.item())

        def savedata(write_disk=False):
            print('-' * 50)
            print('Step: ', i)
            if write_disk:
                pickle.dump(
                    dist_cate,
                    open(os.path.join(args.folder, 'chamfer_dist.pkl'), 'wb'))
                pickle.dump(F_score,
                            open(os.path.join(args.folder, 'f.pkl'), 'wb'))
            print('==> chamfer')
            for c in cates:
                if len(dist_cate[c]) == 0:
                    continue
                print('%s: %.10f' % (c, np.mean(dist_cate[c])))
            print('Mean of ALL: %.10f' % (np.mean([
                np.mean(dist_cate[c]) for c in cates if len(dist_cate[c]) > 0
            ])))
            print('==> F')
            mean_score_list = []
            for c in cates:
                if len(dist_cate[c]) == 0:
                    continue
                print(c, end='')
                s = F_score[c]
                s = np.asarray(s)
                mean_s = np.mean(s, axis=0)
                mean_score_list.append(mean_s)
                for i_tao in range(len(toa)):
                    print(' %.10f' % mean_s[i_tao], end='')
                print('')
            all_mean = np.asarray(mean_score_list)
            all_mean = np.mean(all_mean, axis=0)
            print('ALL Mean:', end='')
            for i_tao in range(len(toa)):
                print(' %.10f' % all_mean[i_tao], end='')
            print('\n')
            return mean_score_list

        if i % 1000 == 999:
            savedata(False)
        # last
        if i == len(meshfiles) - 1:
            mean_s = savedata(True)
            for j, c in enumerate(cates):
                cnew = '%s-chamfer_%.5f-F_%.5f_%.5f' % (c, np.mean(
                    dist_cate[c]), mean_s[j][0], mean_s[j][-1])
                cmd = 'echo 0 > %s/%s.txt' % (folder, cnew)
                print(cmd)
                os.system(cmd)


if __name__ == '__main__':
    main()
