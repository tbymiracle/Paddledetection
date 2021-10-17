# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec
from .name_adapter import NameAdapter

from .resnet import BasicBlock, ConvNormLayer
from .resnet import BottleNeck as _Bottleneck
from .resnet import ResNet

__all__ = ['DetectoRS_ResNet']

ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class BottleNeck(_Bottleneck):
    """Bottleneck for the ResNet backbone in `DetectoRS
    <https://arxiv.org/pdf/2006.02334.pdf>`_.

    This bottleneck allows the users to specify whether to use
    SAC (Switchable Atrous Convolution) and RFP (Recursive Feature Pyramid).

    Args:
         ch_in (int): The number of input channels.
         ch_out (int): The number of output channels before expansion.
         rfp_in_ch (int, optional): The number of channels from RFP.
             Default: None. If specified, an additional conv layer will be
             added for ``rfp_feat``. Otherwise, the structure is the same as
             base class.
         sac (dict, optional): Dictionary to construct SAC. Default: None.
    """
    expansion = 4

    def __init__(self,
                 ch_in,
                 ch_out,
                 rfp_in_ch=None,
                 conv_aws =False,
                 sac=None,
                 **kwargs):
        super(BottleNeck, self).__init__(ch_in, ch_out, **kwargs)

        assert sac is None or isinstance(sac, dict)
        self.sac = sac
        self.conv_aws=conv_aws
        self.with_sac = sac is not None
        if self.with_sac:
            self.branch2b = ConvNormLayer(
                ch_in=self.width,
                ch_out=self.width,
                filter_size=3,
                stride=self.stride2,
                groups=self.groups,
                act='relu',
                norm_type=self.norm_type,
                norm_decay=self.norm_decay,
                freeze_norm=self.freeze_norm,
                lr=self.lr,
                conv_aws =self.conv_aws,
                dcn_v2=self.dcn_v2,
                sac=self.sac)
        # 这里好像是改branch2b，但是mm的是分为conv2和norm2的，ppdet合并了
            

        self.rfp_in_ch = rfp_in_ch
        if self.rfp_in_ch:
            self.rfp_conv = nn.Conv2D(
                self.rfp_in_ch,
                ch_out * self.expansion,
                1,
                stride=1,
                bias_attr=True)

        

    def rfp_forward(self, inputs, rfp_feat):
        """The forward function that also takes the RFP features as input."""
        out = self.branch2a(inputs)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.std_senet:
            out = self.se(out)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        out = paddle.add(x=out, y=short)

        if self.rfp_in_ch:
            rfp_feat = self.rfp_conv(rfp_feat)
            out = paddle.add(x=out, y=rfp_feat)

        out = F.relu(out)

        return out


class Blocks(nn.Layer):
    def __init__(self,
                 block,
                 ch_in,
                 ch_out,
                 count,
                 name_adapter,
                 stage_num,
                 variant='b',
                 groups=1,
                 base_width=64,
                 lr=1.0,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 dcn_v2=False,
                 std_senet=False,
                 conv_aws =False,
                 sac =None,
                 rfp_in_ch=None,
                 **kwargs):
        super(Blocks, self).__init__()

        self.blocks = []
        conv_name = name_adapter.fix_layer_warp_name(stage_num, count, 0)
        layer = self.add_sublayer(
            conv_name,
            block(
                ch_in=ch_in,
                ch_out=ch_out,
                stride=2 if stage_num != 2 else 1,
                shortcut=False,
                variant=variant,
                groups=groups,
                base_width=base_width,
                lr=lr,
                norm_type=norm_type,
                norm_decay=norm_decay,
                freeze_norm=freeze_norm,
                dcn_v2=dcn_v2,
                std_senet=std_senet,
                conv_aws = conv_aws,
                rfp_in_ch = rfp_in_ch,
                sac = sac))
        self.blocks.append(layer)
        ch_in = ch_out * block.expansion
        for i in range(1, count):
            conv_name = name_adapter.fix_layer_warp_name(stage_num, count, i)
            layer = self.add_sublayer(
                conv_name,
                block(
                    ch_in=ch_in,
                    ch_out=ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1,
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    groups=groups,
                    base_width=base_width,
                    lr=lr,
                    norm_type=norm_type,
                    norm_decay=norm_decay,
                    freeze_norm=freeze_norm,
                    dcn_v2=dcn_v2,
                    std_senet=std_senet,
                    conv_aws = conv_aws,
                    sac = sac))
            self.blocks.append(layer)
                

    def forward(self, inputs):
        block_out = inputs
        for block in self.blocks:
            block_out = block(block_out)
        return block_out

    def rfp_forward(self, inputs, rfp_feat):
        block_out = inputs
        for block in self.blocks:
            block_out = block.rfp_forward(block_out,rfp_feat)
        return block_out


@register
@serializable
class DetectoRS_ResNet(ResNet):

    def __init__(self,
                 sac=None,
                 conv_aws =False,
                 stage_with_sac=[False, False, False, False],
                 rfp_in_ch=None,
                 output_img=False,
                 norm_type='bn',
                 **kwargs):
        self.sac = sac
        self.conv_aws = conv_aws
        self.stage_with_sac = stage_with_sac
        self.rfp_in_ch = rfp_in_ch
        self.output_img = output_img
        super(DetectoRS_ResNet, self).__init__(**kwargs,norm_type=norm_type)

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if self.depth >= 50 else BasicBlock

        self._out_channels = [block.expansion * v for v in ch_out_list]
        self._out_strides = [4, 8, 16, 32]

        block_nums = ResNet_cfg[self.depth]
        na = NameAdapter(self)
        self.res_layers = []
        self.ch_in = self.ch_in_org
        for i in range(self.num_stages):

            sac = self.sac if self.stage_with_sac[i] else None
            stage_num = i + 2
            res_name = "res{}".format(stage_num)
            res_layer = self.add_sublayer(
                res_name,
                Blocks(
                    block,
                    self.ch_in,
                    ch_out_list[i],
                    count=block_nums[i],
                    name_adapter=na,
                    stage_num=stage_num,
                    variant=self.variant,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_type=self.norm_type,
                    norm_decay=self.norm_decay,
                    freeze_norm=self.freeze_norm,
                    dcn_v2=(i in self.dcn_v2_stages),
                    std_senet=self.std_senet,
                    sac=sac,
                    conv_aws = conv_aws,
                    rfp_in_ch=rfp_in_ch if i > 0 else None))
            self.res_layers.append(res_layer)
            self.ch_in = self._out_channels[i]

        if self.freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(self.freeze_at + 1, self.num_stages)):
                self._freeze_parameters(self.res_layers[i])

    def forward(self, x):
        """Forward function."""
        outs = list(super(DetectoRS_ResNet, self).forward(x))
        if self.output_img:
            outs.insert(0, x)
        return outs


    def rfp_forward(self, x, rfp_feats):
        """Forward function for RFP."""
        x = x['image']
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            rfp_feat = rfp_feats[i] if i > 0 else None
            x = res_layer.rfp_forward(x, rfp_feat)
            if i in self.return_idx:
                outs.append(x)
        return outs
