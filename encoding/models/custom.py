from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

from GIFNet_model import TwoBranchesFusionNet  # Import the GIFNet fusion network

__all__ = ['DeepLabV3', 'get_deeplab']

class DeepLabV3(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DeepLabV3, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.fusion_net = TwoBranchesFusionNet(s=3, n=64, channel=3, stride=1)  # Initialize GIFNet fusion network

        self.head = DeepLabV3Head(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

#     def forward(self, x):

# # Process input through the fusion network
#         fused_features = self.fusion_net(x[:, :3, :, :], x[:, 3:, :, :])  # Assuming first 3 channels are one modality and the next are another

#         # fused_features = self.fusion_net(x, x)

#         # _, _, h, w = x.size()
#         # _, _, c3, c4 = self.base_forward(x)
#         _, _, h, w = x[:, :3, :, :].size()
#         _, _, c3, c4 = self.base_forward(fused_features)  # Pass fused features to DeepLabV3 backbone

    def forward(self, ir_img, vis_img):
        """前向传播，接受两个独立的输入张量
        
        Args:
            ir_img (torch.Tensor): 红外图像张量，形状 [B, 3, H, W]
            vis_img (torch.Tensor): 可见光图像张量，形状 [B, 3, H, W]
        
        Returns:
            tuple: 包含分割结果的元组
        """
        # 通过融合网络处理双模态输入
        fused_features = self.fusion_net(ir_img, vis_img)
        
        # 获取输入尺寸
        _, _, h, w = ir_img.size()
        
        # 将融合特征传入基础网络
        _, _, c3, c4 = self.base_forward(fused_features)


        outputs = []
        x = self.head(c4)
        x = F.interpolate(x, (h,w), **self._up_kwargs)
        
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)


class DeepLabV3Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates=(12, 24, 36)):
        super(DeepLabV3Head, self).__init__()
        inter_channels = in_channels // 8
        self.aspp = ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        return x


def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block

class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h,w), **self._up_kwargs)

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)

        return self.project(y)


def get_deeplab(dataset='custom', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = DeepLabV3(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
