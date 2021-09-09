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
from torch.utils.data import Dataset, DataLoader

import sys

sys.path.append('..')
from dataloader.dataloader_dvr import DataProvider, collate_fn


####################
# Load the dataset #
####################
def get_data_loaders(imfolder, classnames, viewnum, mode, bs, num_workers):

    print('Building dataloaders')

    dataset_train = DataProvider(imfolder, classnames, viewnum, mode=mode)

    shuffle = True
    if mode == 'train_val' or mode == 'test':
        shuffle = False

    train_loader = DataLoader(dataset_train, batch_size=bs, \
                              shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    print('train num {}'.format(len(dataset_train)))
    print('train iter'.format(len(train_loader)))

    return train_loader


#######################################################
# Load the dataset from args
#######################################################
def prepare_dataloader(args, mode):
    '''
    return a pytorch dataloader for shapenet dataset
    using dataset from https://github.com/autonomousvision/differentiable_volumetric_rendering
    '''

    # use all the 13 shapenet classes
    cls = [
        '02691156', '02828884', '02933112', '02958343', '03001627', '03211117',
        '03636649', '03691459', '04090263', '04256520', '04379243', '04401088',
        '04530566'
    ]
    dataloader = get_data_loaders(args.datafolder, cls, args.viewnum, \
                            mode=mode, \
                            bs=args.batch_size, num_workers=args.num_workers)

    return dataloader, cls


#######################################################
# parse the data batch
#######################################################
def parse_dataloader(data_batch, viewnum, device):
    '''
    parse the batched data and return images and viewpoints
    '''
    gtims = []
    camrots = []
    camposes = []
    for j in range(viewnum):

        im_bxhxwx3 = data_batch['view%d' % j]['im224']
        im_bx4xhxw = im_bxhxwx3.permute(0, 3, 1, 2).to(device)
        camrot_bx3x3 = data_batch['view%d' % j]['camrot'].to(device)
        campos_bx3 = data_batch['view%d' % j]['campos'].to(device)

        gtims.append(im_bx4xhxw)
        camrots.append(camrot_bx3x3)
        camposes.append(campos_bx3)

    return gtims, camrots, camposes


if __name__ == '__main__':

    #################################################
    imszs = [256]
    viewnum = 2

    folder = '/home/wenzheng/largestore/data-generated/dvr/differentiable_volumetric_rendering/data/NMR_Dataset'
    folder = '/home/wz/dataset/dibr/NMR_Dataset'

    split = 'softras'
    cates = '02691156,02828884,02933112,02958343,03001627,03211117,03636649,03691459,04090263,04256520,04379243,04401088,04530566'
    cates = cates.split(',')

    ####################################
    train_loader = get_data_loaders(folder, cates, viewnum, \
                                    mode='train', \
                                    bs=32, num_workers=8)

    ##############################################
    for i, data in enumerate(train_loader):
        if data is None:
            continue
        for j in range(viewnum):
            for key in ['im224', 'camrot', 'campos', 'viewidx']:
                print('{}, view{}, {}, {}, {}'.format(
                    i, j, key, data['view%d' % j][key].shape,
                    data['view%d' % j][key].dtype))

        # for key in ['cate', 'md5']:
        #   print('{}, {}, {}'.format(i, key, data[key]))
