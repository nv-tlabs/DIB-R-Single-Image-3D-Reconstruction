'''
MIT License

Copyright (c) 2020 Autonomous Vision Group (AVG),  Max Planck Institute for Intelligent Systems Tübingen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

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
'''
The functions in file is mostly borrowed from
https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/11542ed5ac4e7e4c19c5c74eba7929c1333f3896/im2mesh/dvr/models/__init__.py
with some modifications.
Codes released under MIT license
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .conv import Resnet18
import numpy as np


########################################################
class DVR(nn.Module):
    ''' DVR model class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        template (torch.FloatTensor): of shape (num_vertices, 3), template mesh
    '''
    def __init__(self, template):
        super(DVR, self).__init__()

        decoder = Decoder(dim=3,
                          c_dim=256,
                          leaky=True,
                          out_dim=6,
                          res0=True,
                          res0ini=torch.ones)

        encoder = Resnet18(c_dim=256, normalize=True, use_linear=True)

        self.decoder = decoder
        self.encoder = encoder

        self.template = nn.Parameter(template, requires_grad=False)

        # learn the delta
        residual_coef = torch.zeros(1)
        self.residual_coef = nn.Parameter(residual_coef)

    def forward(self, inputs_bx3xhxw):

        # encode inputs
        c_bxc = self.encoder(inputs_bx3xhxw)

        pred_bxpxk = self.decoder(self.template, c=c_bxc)
        rgb = pred_bxpxk[:, :, :3]
        rgb = F.sigmoid(rgb)

        delta = pred_bxpxk[:, :, 3:6]

        p = self.template + self.residual_coef * delta

        return p, delta, rgb
