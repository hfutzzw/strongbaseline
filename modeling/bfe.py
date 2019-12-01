# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import random

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.osnet import OSNet, OSBlock

import pdb


class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask

        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        try:
            if m.bias.size():
                # pdb.set_trace()
                nn.init.constant_(m.bias, 0.0)
        except:
            if m.bias:
                nn.init.constant_(m.bias, 0.0)


class BFE(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(BFE, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],  # number of inserted SEblocks between each two resnet convblocks
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        elif model_name == 'osnet':
            self.in_planes = 512
            self.base = OSNet(blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                              channels=[64, 256, 384, 512], IN=True)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.h_ratio = 0.33
        self.w_ratio = 1.0

        self.batchcrop = BatchDrop(self.h_ratio,self.w_ratio)

        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.reduct_dim = self.in_planes // 2

        # bottleneck for
        self.bottleneck = Bottleneck(self.in_planes,self.in_planes//4)

        # reduct channel dimension for local branch
        self.reduction = nn.Sequential(
            nn.Conv2d(self.in_planes, self.reduct_dim, 1),
            nn.BatchNorm2d(self.reduct_dim),
            nn.ReLU()
        )

        if self.neck == 'no':
            self.classifier_global = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck_global = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.classifier_global = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck_local = nn.BatchNorm1d(self.reduct_dim)
            self.bottleneck_local.bias.requires_grad_(False)
            self.classifier_local = nn.Linear(self.reduct_dim,self.num_classes,bias=False)

            self.bottleneck_global.apply(weights_init_kaiming)
            self.classifier_global.apply(weights_init_classifier)
            self.bottleneck_local.apply(weights_init_kaiming)
            self.classifier_local.apply(weights_init_classifier)

    def forward(self, x):
        featuremap = self.base(x)  # (b,2048,h,w)
        local_featuremap = self.bottleneck(featuremap)  #(b,2048,h,w)
        local_featuremap = self.batchcrop(local_featuremap)  #(b,2048,h,w)

        global_feat = self.gap(featuremap)   # (b,2048, 1, 1)
        local_feat = self.gmp(local_featuremap)    # (b,2048,1,1)
        local_feat = self.reduction(local_feat)   #(b,1024,1,1)

        g_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        l_feat = local_feat.view(local_feat.shape[0],-1)   # flattten to (bs,2048)

        if self.neck == 'bnneck':
            global_feat = self.bottleneck_global(g_feat)  # normalize for angular softmax
            local_feat = self.bottleneck_local(l_feat)

        if self.training:
            if self.neck == 'bnneck':
                cls_score_global = self.classifier_global(global_feat)
                cls_score_local = self.classifier_local(local_feat)
            else:
                cls_score_global = self.classifier_global(g_feat)
                cls_score_local = self.classifier_local(l_feat)

            return [cls_score_global,cls_score_local], [g_feat,l_feat]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat((global_feat,local_feat),1)
            else:
                # print("Test with feature before BN")
                return torch.cat((g_feat,l_feat),1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        # pdb.set_trace()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
