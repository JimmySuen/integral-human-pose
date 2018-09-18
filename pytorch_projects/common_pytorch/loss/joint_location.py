import numpy as np
from easydict import EasyDict as edict

import torch.nn as nn

from common_pytorch.common_loss.weighted_mse import weighted_l1_loss, weighted_mse_loss


# config
def get_default_loss_config():
    config = edict()
    config.loss_type = 'L2'
    return config


# config

# define loss
def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class L2JointLocationLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(L2JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]
        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_mse_loss(preds, gt_joints, gt_joints_vis, self.size_average)


class L1JointLocationLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(L1JointLocationLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]
        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_l1_loss(preds, gt_joints, gt_joints_vis, self.size_average)


def get_loss_func(config):
    if config.loss_type == 'L1':
        return L1JointLocationLoss()
    elif config.loss_type == 'L2':
        return L2JointLocationLoss()
    else:
        assert 0, 'Error. Unknown heatmap type {}'.format(config.heatmap_type)


# define loss


# define label
def generate_joint_location_label(config, patch_width, patch_height, joints, joints_vis):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width
    joints = joints.reshape((-1))
    joints_vis = joints_vis.reshape((-1))
    return joints, joints_vis


def get_label_func(config):
    return generate_joint_location_label


# define label


# define result
def get_joint_location_result(config, patch_width, patch_height, preds):
    coords = preds.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 3), 3))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height
    coords[:, :, 2] = coords[:, :, 2] * patch_width
    scores = np.ones((coords.shape[0], coords.shape[1], 1), dtype=float)
    coords = np.concatenate((coords, scores), axis=2)
    return coords


def get_result_func(config):
    return get_joint_location_result


# define result

# define merge
def merge_flip_func(a, b, flip_pair):
    return a


def get_merge_func(loss_config):
    return merge_flip_func
# define merge
