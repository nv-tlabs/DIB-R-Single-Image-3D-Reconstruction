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

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='dibr')

    # dataloader
    parser.add_argument('--datafolder',
                        type=str,
                        default='YOUR DATASET FOLDER',
                        help='dataset folder')

    parser.add_argument('--num-workers',
                        type=int,
                        default=8,
                        help='num_workers term in pytorch dataloader')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='batch size')
    parser.add_argument(
        '--viewnum',
        type=int,
        default=2,
        help=
        'how many views do we use for one 3D model, 2 is enough for training, 1 for test'
    )

    # save folder
    parser.add_argument('--svfolder',
                        type=str,
                        default='YOUR SAVING FOLDER',
                        help='save folder')

    # training settings
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--lrdecay', type=float, default=0.5, help='lrdecay')
    parser.add_argument('--decaystep',
                        type=float,
                        default=400000,
                        help='how many iterations to decay lr')

    parser.add_argument(
        '--iterbe',
        type=int,
        default=-1,
        help=
        'start iteration, -1 means starting from strach, others means starting from pretrained model'
    )
    parser.add_argument('--epoch',
                        type=int,
                        default=600,
                        help='training epoch')

    args = parser.parse_args()

    return args
