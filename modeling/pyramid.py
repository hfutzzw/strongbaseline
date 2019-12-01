import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from modeling.backbones.resnet_ibn_a import  resnet50_ibn_a
import random

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

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        # init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        # init.constant(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        # add_block = []
        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck, bias=False)]
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]

        # add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_kaiming)
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block1 = add_block1
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block1(x)
        x1 = self.add_block2(x)
        x2 = self.classifier(x1)
        return x, x1, x2

# ft_net_50_1
class Pyramid(nn.Module):

    def __init__(self, class_num):
        super(Pyramid, self).__init__()

        #model_ft = models.resnet50(pretrained=True)

        model_ft = resnet50_ibn_a(last_stride=1)
        model_ft.load_param('/home/zzw/re_id/strongbaseline/pretrained_models/resnet50_ibn_a.pth.tar')

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool_2 = nn.AdaptiveAvgPool2d((2,2))

        self.avgpool_2 = nn.AdaptiveAvgPool2d((2, 2))
        self.avgpool_3 = nn.AdaptiveMaxPool2d((2, 2))
        self.avgpool_4 = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool_5 = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier_1 = ClassBlock(1024, class_num, num_bottleneck=512)

        self.classifier_2 = ClassBlock(2048, class_num, num_bottleneck=512)
        self.classifier_3 = ClassBlock(8192, class_num, num_bottleneck=512)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)

        x0 = self.model.layer3(x)
        x = self.model.layer4(x0)

        # global average and max pooling

        x3 = self.model.avgpool(x)
        #x3 = self.avgpool_1(x)
        x_3 = self.avgpool_5(x)

        # local average and max pooling
        x_41 = self.avgpool_2(x)
        x_4 = self.avgpool_3(x)

        # multi-scale global average and max pooling
        x_0 = self.avgpool_1(x0)
        x_1 = self.avgpool_4(x0)

        x0 = x_0 + x_1
        x_31 = x3 + x_3
        x4 = x_41 + x_4
        #
        x6 = torch.squeeze(x0)
        x_0 = torch.squeeze(x_0)
        x_1 = torch.squeeze(x_1)

        x3 = torch.squeeze(x3)
        x_3 = torch.squeeze(x_3)
        # x7 = x1.view(x1.size(0),-1)

        #
        x9 = torch.squeeze(x_31)
        x_10 = x_4.view(x_4.size(0), -1)
        x_11 = x_41.view(x_41.size(0), -1)
        x10 = x4.view(x4.size(0), -1)

        #
        x14, x15, x16 = self.classifier_1(x6)
        x19, x17, x18 = self.classifier_2(x9)
        x23, x21, x22 = self.classifier_3(x10)


        if self.training:
            return [x16, x18, x22], [x_0, x_1, x3, x_3, x_10, x_11]
        else:
            #return torch.cat((x15,x17,x21),1)
            return torch.cat((x14,x19,x23,x_0,x_1),1)

            #pre_norm1 = torch.cat((x_0,x_1),1)
            #pre_norm1 = torch.nn.functional.normalize(pre_norm1, dim=1, p=2)

            #pre_norm2 = torch.cat((x14,x19,x23),1)
            #pre_norm2 = torch.nn.functional.normalize(pre_norm2, dim=1, p=2)

            #return torch.cat((pre_norm1,pre_norm2),1)
