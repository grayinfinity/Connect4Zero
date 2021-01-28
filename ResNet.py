import torch
import torch.utils
import torch.nn as nn
import torch.torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import time
import math
from collections import OrderedDict
import torch.utils.data
import numpy as np
import config
import random


# ================================= CLASS : basic ResNet Block ================================= #

# no bias in conv
def conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = torch.nn.BatchNorm2d(planes)
        m['relu1'] = torch.nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = torch.nn.BatchNorm2d(planes)
        self.group1 = torch.nn.Sequential(m)

        self.relu = torch.nn.Sequential(torch.nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out


# ================================= CLASS : ResNet + two heads ================================= #

class ResNet(torch.nn.Module):
    def __init__(self, block, layers):

        self.input_dim = 7 * 6
        self.output_dim = 7
        self.inplanes = 128
        self.convsize = 128
        super(ResNet, self).__init__()

        torch.set_num_threads(1)

        # as a start : the three features are mapped into a conv with 4*4 kernel
        self.ksize = (4, 4)
        self.padding = (1, 1)
        m = OrderedDict()
        m['conv1'] = torch.nn.Conv2d(3, self.convsize, kernel_size=self.ksize, stride=1, padding=self.padding, bias=False)
        m['bn1'] = torch.nn.BatchNorm2d(self.convsize)
        m['relu1'] = torch.nn.ReLU(inplace=True)

        self.group1 = torch.nn.Sequential(m)

        # next : entering the resnet tower
        self.layer1 = self._make_layer(block, self.convsize, layers[0])

        # next : entering the policy head
        pol_filters = 2
        self.policy_entrance = torch.nn.Conv2d(self.convsize, 2, kernel_size=1, stride=1, padding=0,
                                         bias=False)
        self.bnpolicy = torch.nn.BatchNorm2d(2)
        self.relu_pol = torch.nn.ReLU(inplace=True)

        self.fcpol2 = torch.nn.Linear(pol_filters * 30, 7)

        self.softmaxpol = torch.nn.Softmax(dim=1)
        # end of policy head

        # in parallel: entering the value head
        val_filters = 1
        self.value_entrance = torch.nn.Conv2d(self.convsize, 1, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.bnvalue = torch.nn.BatchNorm2d(1)
        self.relu_val = torch.nn.ReLU(inplace=True)

        # entering a dense hidden layer
        self.hidden_dense_value = torch.nn.Linear(val_filters * 30, 256)
        self.relu_hidden_val = torch.nn.ReLU(inplace=True)
        self.fcval = torch.nn.Linear(256, 1)
        self.qval = torch.nn.Tanh()
        # end value head

        # init weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / (5 * n)))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = x.reshape((3, 6, 7))
            x = torch.FloatTensor(x)
            x = torch.unsqueeze(x, 0)

        x = x.cuda()
        x = self.group1(x)
        x = self.layer1(x)

        x1 = self.policy_entrance(x)
        x1 = self.bnpolicy(x1)
        x1 = self.relu_pol(x1)
        x1 = x1.view(-1, 2 * 30)

        x1 = self.fcpol2(x1)

        x1 = self.softmaxpol(x1)

        x2 = self.value_entrance(x)
        x2 = self.bnvalue(x2)
        x2 = self.relu_val(x2)
        x2 = x2.view(-1, 30 * 1)
        x2 = self.hidden_dense_value(x2)
        x2 = self.relu_hidden_val(x2)
        x2 = self.fcval(x2)
        x2 = self.qval(x2)

        return x2, x1


# -----------------------------------------------------------------#
# builds the model
def resnet18(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [20, 2, 2, 2], **kwargs)
    model.eval()
    return model


# ================================= CLASS : ResNet training ================================= #

