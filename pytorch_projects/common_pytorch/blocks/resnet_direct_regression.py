import os
from easydict import EasyDict as edict

import torch
import torch.nn as nn
from torchvision.models.resnet import model_zoo, model_urls

from common_pytorch.base_modules.resnet import resnet_spec, ResNetBackbone
from common_pytorch.base_modules.avg_pool_head import AvgPoolHead


def get_default_network_config():
    config = edict()
    config.from_model_zoo = True
    config.pretrained = ''
    config.num_layers = 50
    config.fea_map_size = 0
    return config


class ResPoseNet(nn.Module):
    def __init__(self, backbone, head):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def get_pose_net(cfg, num_joints):
    block_type, layers, channels, name = resnet_spec[cfg.num_layers]
    backbone_net = ResNetBackbone(block_type, layers)
    head_net = AvgPoolHead(channels[-1], num_joints * 3, cfg.fea_map_size)
    pose_net = ResPoseNet(backbone_net, head_net)
    return pose_net


def init_pose_net(pose_net, cfg):
    if cfg.from_model_zoo:
        _, _, _, name = resnet_spec[cfg.num_layers]
        org_resnet = model_zoo.load_url(model_urls[name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        pose_net.backbone.load_state_dict(org_resnet)
        # print('Loading pretrained model from {}'.format(os.path.join(cfg.pretrained, model_file)))
    else:
        if os.path.exists(cfg.pretrained):
            model = torch.load(cfg.pretrained)
            pose_net.load_state_dict(model['network'])
            print("Init Network from pretrained", cfg.pretrained)
