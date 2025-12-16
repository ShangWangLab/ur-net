"""Model definitions and data loading utility functions.

MicroNet is the final implementation of µ-Net.
MicroRNet is the final implementation of the refinement µ-Net (µr-Net).
"""

import os

import nrrd
import numpy as np
import torch
import torch.nn as nn


def lecun_init(tensor):
    fan_in = tensor[0].numel()
    std = fan_in**-0.5
    with torch.no_grad():
        tensor.normal_(0, std)


class MicroRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.name = "ur-net"
        self.down0 = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.AvgPool3d(4),
            nn.Conv3d(5, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
        )
        self.bottom = nn.Sequential(
            nn.AvgPool3d(4),
            nn.Conv3d(4, 3, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(3, 3, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
            nn.Conv3d(3, 3, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(3, 3, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv3d(7, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
        )
        self.up0 = nn.Sequential(
            nn.Conv3d(9, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 1, 3, padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                lecun_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        struct = x[:, 0, None, :, :, :]
        mask = x[:, 1, None, :, :, :]
        x0 = torch.cat((mask, self.down0(struct)), 1)
        x1 = self.down1(x0)
        x2 = self.bottom(x1)
        x2r = nn.Upsample(x1.shape[2:])(x2)
        x3 = self.up1(torch.cat((x1, x2r), 1))
        x3r = nn.Upsample(x0.shape[2:])(x3)
        x4 = self.up0(torch.cat((x0, x3r), 1))
        return x4


class MicroNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.name = "u-net"
        self.down0 = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.AvgPool3d(4),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
        )
        self.bottom = nn.Sequential(
            nn.AvgPool3d(4),
            nn.Conv3d(4, 3, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(3, 3, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
            nn.Conv3d(3, 3, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(3, 3, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv3d(7, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
        )
        self.up0 = nn.Sequential(
            nn.Conv3d(8, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.SELU(inplace=True),
            nn.Conv3d(4, 4, 3, padding=1, padding_mode="reflect"),
            nn.Conv3d(4, 1, 3, padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                lecun_init(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.bottom(x1)
        x2r = nn.Upsample(x1.shape[2:])(x2)
        x3 = self.up1(torch.cat((x1, x2r), 1))
        x3r = nn.Upsample(x0.shape[2:])(x3)
        x4 = self.up0(torch.cat((x0, x3r), 1))
        return x4


# These 3D images have been scaled down to isometric 4.5 microns/pixel.
def load_struct(name):
    return nrrd.read("struct_iso/" + name + ".nrrd", index_order="C")


def load_img(name, device="cpu"):
    img, header = load_struct(name)
    img = torch.tensor(img, dtype=torch.float32, device=device)
    img -= img.mean()
    img /= img.std()
    return img, header


def load_annotation(name):
    seg, _ = nrrd.read("annotations/" + name + ".seg.nrrd", index_order="C")
    return seg[:, :, :, 1]
