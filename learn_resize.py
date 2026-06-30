
# Learn to Resize

# https://github.com/yundaehyuck/Learning-to-resize-images-for-computer-vision-tasks/blob/main/resizing_network.ipynb

import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_channels=16):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


def make_block(r, n):
    residual = []

    for i in range(r):
        block = ResBlock(num_channels=n)
        residual.append(block)

    return nn.Sequential(*residual)

# Simple resize to compare with learnable resize
class FixedResizeNetwork(nn.Module):
    def __init__(self):
        super(FixedResizeNetwork, self).__init__()
    def forward(self, x):
        # out = F.interpolate(x, scale_factor=0.5, mode='linear', align_corners=False,recompute_scale_factor=False)
        out = F.interpolate(x, scale_factor=0.5, mode='area') #, align_corners=False,recompute_scale_factor=False)
        return out

# Learnable resize
class ResizingNetwork(nn.Module):
    def __init__(self, r=1, n=16, in_channels=1, simple_resize=False):
        super(ResizingNetwork, self).__init__()

        self.simple_resize = simple_resize

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n, kernel_size=7, stride=1, padding=3)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(n, n, kernel_size=1, stride=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(n)

        self.resblock = make_block(r, n)

        self.conv3 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

        self.conv4 = nn.Conv2d(n, out_channels=3, kernel_size=7, stride=1, padding=3)


    def forward(self, x):

        residual = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False,recompute_scale_factor=False)

        if self.simple_resize:
            return residual

        out = self.conv1(x)
        out = self.leakyrelu1(out)

        out = self.conv2(out)
        out = self.leakyrelu2(out)
        out = self.bn1(out)

        out_residual = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False,recompute_scale_factor=False)

        out = self.resblock(out_residual)

        out = self.conv3(out)
        out = self.bn2(out)
        out += out_residual

        out = self.conv4(out)
        out += residual

        return out