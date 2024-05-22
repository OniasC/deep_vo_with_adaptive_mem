import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np

import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_

from convlstm import ConvLSTM

# Creating the model of our Deep Visual Odometry

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

class DvoAm_Encoder(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(DvoAm_Encoder, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)

        # to say that i dont need to train these after initialzing
        for param in self.conv1[0].parameters(): #this covers weights and biases
            param.requires_grad = False
        for param in self.conv2[0].parameters():
            param.requires_grad = False
        for param in self.conv3[0].parameters():
            param.requires_grad = False
        for param in self.conv3_1[0].parameters():
            param.requires_grad = False
        for param in self.conv4[0].parameters():
            param.requires_grad = False
        for param in self.conv4_1[0].parameters():
            param.requires_grad = False
        for param in self.conv5[0].parameters():
            param.requires_grad = False
        for param in self.conv5_1[0].parameters():
            param.requires_grad = False
        for param in self.conv6[0].parameters():
            param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv5(self.conv6(out_conv5))
        return out_conv6


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]

class DvoAm_EncPlusTrack(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(DvoAm_EncPlusTrack, self).__init__()

        self.batchNorm = batchNorm
        # entrada do tensor das duas imagens H x W x 6 (2 rgb's)
        self.encoding = DvoAm_Encoder(batchNorm)
        # para entrada de 640 x 480 x 6, a saida eh de 10 x 8 x 1024
        self.convLSTM = ConvLSTM(input_dim = 6, hidden_dim = 1, kernel_size = (3,3), num_layers=1)
        # convnLTSM nao muda tamanho do tensor
        self.gap = nn.AdaptiveMaxPool3d((1,1,1024))
        # (1,1,1024)
        self.fc = nn.Linear(in_features=1024,out_features=6) # should this be 7 because 3 position + 4 quaternions?
        # a saida eh 6
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_encoder = self.encoding(x)
        out_tracking = self.fc(self.gap(self.convLSTM(out_encoder)))
        return out_tracking