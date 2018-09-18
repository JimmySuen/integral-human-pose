import random
import numpy as np
from easydict import EasyDict as edict

def get_default_augment_config():
    config = edict()
    config.scale_factor = 0.25
    config.rot_factor = 30
    config.color_factor = 0.2
    config.do_flip_aug = True

    config.rot_aug_rate = 0.6  #possibility to rot aug
    config.flip_aug_rate = 0.5 #possibility to flip aug
    return config

def do_augmentation(aug_config):
    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * aug_config.rot_factor if random.random() <= aug_config.rot_aug_rate else 0
    do_flip = aug_config.do_flip_aug and random.random() <= aug_config.flip_aug_rate
    c_up = 1.0 + aug_config.color_factor
    c_low = 1.0 - aug_config.color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, color_scale