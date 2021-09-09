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

import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


##############################################################
# load data split
def loadlist(filename, prefix):
    pkl_list = []
    with open(filename, 'r') as f:
        while (True):
            line = f.readline().strip()
            if not line:
                break
            line = '%s/%s' % (prefix, line)
            pkl_list.append(line)
    return pkl_list


#######################################################
class DataProvider(Dataset):
    r'''
    Class for the data provider
    '''

    def __init__(self, \
                 datafolder,
                 cls,
                 viewnum, \
                 imszs=[224], \
                 split='softras',
                 mode='train'):
        r'''
        Args:
            datafolder: NMR dataset folder
            cls: 13 shapenet clasess
            viewnum: how many views do we use
            imszs: image size of the network
            split: dataset split, use softras split
            mode: 'train' or 'test'
        '''

        self.mode = mode
        self.imszs = imszs
        self.viewum = viewnum
        assert self.viewum >= 1

        self.folder = datafolder
        self.pkl_list = []
        for cl in cls:
            splitfile = '%s/%s/%s_%s.lst' % (datafolder, cl, split, mode)
            self.pkl_list.extend(
                loadlist(splitfile, '%s/%s' % (datafolder, cl)))

        self.imnum = len(self.pkl_list)
        print(self.pkl_list[0])
        print(self.pkl_list[-1])
        print('imnum {}'.format(self.imnum))

    def __len__(self):
        return self.imnum

    def __getitem__(self, idx):
        return self.prepare_instance(idx)

    def load_im_cam(self, pkl_path, catagory, md5name, num):

        imname = '%s/image/%04d.png' % (pkl_path, num)
        im = cv2.imread(imname, 1).astype('float32') / 255.0
        immaskname = '%s/mask/%04d.png' % (pkl_path, num)
        mask = cv2.imread(immaskname, 0).astype('float32') / 255.0

        ims = []
        for imsz in self.imszs:
            if im.shape[0] != imsz:
                # we need to resize 64 to 224
                # generally height == width
                h1 = im.shape[0]
                h2 = imsz
                while h1 / h2 > 2 or h1 / h2 < 0.5:
                    if h1 > h2:
                        im = cv2.resize(im, (h1 // 2, h1 // 2),
                                        interpolation=cv2.INTER_AREA)
                        mask = cv2.resize(mask, (h1 // 2, h1 // 2),
                                          interpolation=cv2.INTER_AREA)
                    else:
                        im = cv2.resize(im, (h1 * 2, h1 * 2),
                                        interpolation=cv2.INTER_AREA)
                        mask = cv2.resize(mask, (h1 * 2, h1 * 2),
                                          interpolation=cv2.INTER_AREA)
                    h1 = im.shape[0]
                im = cv2.resize(im, (h2, h2), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (h2, h2), interpolation=cv2.INTER_AREA)

            im_hxwx4 = np.concatenate([im, mask[:, :, None]], axis=2)
            ims.append(im_hxwx4)

        rotntxname = '%s/cameras.npz' % (pkl_path)
        cam = np.load(rotntxname)
        camera_mat = cam['camera_mat_%d' % num]
        world_mat = cam['world_mat_%d' % num]
        camera_mat_inv = cam['camera_mat_inv_%d' % num]
        world_mat_inv = cam['world_mat_inv_%d' % num]

        rotmtx_4x4 = world_mat
        rotmx = rotmtx_4x4[:3, :3]
        transmtx = rotmtx_4x4[:3, 3:4]

        transmtx = -np.matmul(rotmx.T, transmtx)
        # for softras dataset
        rotmx = np.matmul(
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32),
            rotmx)
        renderparam = (rotmx.astype(np.float32), transmtx.astype(np.float32))

        return ims, renderparam

    def prepare_instance(self, idx):
        r'''
        Prepare a single instance
        '''

        re = {}
        re['valid'] = True

        # name parameters
        pkl_path = self.pkl_list[idx]
        fname, md5name = os.path.split(pkl_path)
        fname, category = os.path.split(fname)
        re['cate'] = category
        re['md5'] = md5name

        try:
            if self.viewum == 1:
                # num = int(numname)
                num = np.random.randint(24)
                if self.mode == 'test':
                    num = 0
                ims, renderparam = self.load_im_cam(pkl_path, category,
                                                    md5name, num)

                i = 0
                re['view%d' % i] = {}
                re['view%d' % i]['camrot'] = renderparam[0]
                re['view%d' % i]['campos'] = renderparam[1]
                re['view%d' % i]['viewidx'] = num
                for imi, imsz in enumerate(self.imszs):
                    re['view%d' % i]['im%d' % imsz] = ims[imi]
            else:
                nums = np.random.permutation(24)[:self.viewum]
                for i, num in enumerate(nums):
                    # 24 views in total
                    ims, renderparam = self.load_im_cam(
                        pkl_path, category, md5name, num)

                    re['view%d' % i] = {}
                    re['view%d' % i]['camrot'] = renderparam[0]
                    re['view%d' % i]['campos'] = renderparam[1][:, 0]
                    re['view%d' % i]['viewidx'] = num
                    for imi, imsz in enumerate(self.imszs):
                        re['view%d' % i]['im%d' % imsz] = ims[imi]
        except:
            re['valid'] = False
            return re

        return re


def collate_fn(batch_list):
    for data in batch_list:
        if not data['valid']:
            # print('{}, {}'.format(data['cate'], data['md5']))
            pass

    collated = {}
    batch_list = [data for data in batch_list if data['valid']]
    if len(batch_list) == 0:
        return None

    # keys = batch_list[0].keys()
    keys = ['cate', 'md5']
    for key in keys:
        val = [item[key] for item in batch_list]
        collated[key] = val

    viewnum = len(batch_list[0].keys()) - 3
    # keys = ['im224', 'camrot', 'campos', 'num']
    keys = batch_list[0]['view0'].keys()
    assert 'viewidx' in keys

    for i in range(viewnum):
        collated['view%d' % i] = {}
        for key in keys:
            val = [item['view%d' % i][key] for item in batch_list]
            val = np.stack(val, axis=0)
            if key == 'viewidx':
                collated['view%d' % i][key] = val
            else:
                collated['view%d' % i][key] = torch.from_numpy(val)

    return collated
