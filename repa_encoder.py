import copy
import torch
import torch.nn as nn
import timm
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import os

class IndentityMapping(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        return x

class DINOv2(nn.Module):
    def __init__(self, weight_path:str=None, base_patch_size=16):
        super(DINOv2, self).__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()
        self.base_patch_size = base_patch_size
        self.encoder.compile()

    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        if resize:
            x = torch.nn.functional.interpolate(x, (int(14*h/self.base_patch_size), int(14*w/self.base_patch_size)), mode='bicubic')
        feature = self.encoder.forward_features(x)['x_norm_patchtokens']
        return feature