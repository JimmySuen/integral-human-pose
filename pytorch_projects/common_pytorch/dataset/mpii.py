from __future__ import print_function

import os

import numpy as np
from numpy import transpose
import pickle as pk
import json

from scipy.io import loadmat, savemat

from common.utility.visualization import debug_vis

from .imdb import IMDB
from common.utility.utils import calc_kpt_bound


def check_config_type(c1, c2):
    return set(c1.keys()) == set(c2.keys())


class mpii(IMDB):

    def __init__(self, image_set_name, dataset_path, patch_width, patch_height, *args):
        '''
        0-R_Ankle, 1-R_Knee, 2-R_Hip, 3-L_Hip, 4-L_Knee, 5-L_Ankle, 6-Pelvis, 7-Thorax,
        8-Neck, 9-Head, 10-R_Wrist, 11-R_Elbow, 12-R_Shoulder, 13-L_Shoulder, 14-L_Elbow, 15-L_Wrist
        '''
        super(mpii, self).__init__('mpii', image_set_name, dataset_path, patch_width, patch_height)

        self.joint_num = 16
        self.flip_pairs = np.array([[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]], dtype=np.int)
        self.parent_ids = np.array([1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14], dtype=np.int)
        self.pixel_std = 200

        self.aspect_ratio = self.patch_width * 1.0 / self.patch_height

    def gt_db(self):

        cache_file = os.path.join(self.cache_path, self.name + '_keypoint_db_v3' + '.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))
            return db

        # create train/val split
        with open(os.path.join(self.dataset_path, 'annot', self.image_set_name + '.json')) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            # center and size
            c = np.array(a['center'], dtype=np.float)
            c_x = c[0]
            c_y = c[1]
            assert c_x >= 1
            c_x = c_x - 1
            c_y = c_y - 1
            s = np.array([a['scale'], a['scale']], dtype=np.float)
            width = s[0]
            height = s[1]
            # Adjust center/scale slightly to avoid cropping limbs, this is the common practice on mpii dataset
            c_y = c_y + 15 * height

            width = width * 1.25 * self.pixel_std
            height = height * 1.25 * self.pixel_std

            if width / height >= 1.0 * self.patch_width / self.patch_height:
                width = 1.0 * height * self.patch_width / self.patch_height
            else:
                assert 0, "Error. Invalid patch width and height"

            # joints and vis
            jts_3d = np.zeros((self.joint_num, 3), dtype=np.float)
            jts_3d_vis = np.zeros((self.joint_num, 3), dtype=np.float)
            if self.image_set_name != 'test':
                jts = np.array(a['joints'])
                jts[:, 0:2] = jts[:, 0:2] - 1
                jts_vis = np.array(a['joints_vis'])
                assert len(jts) == self.joint_num, 'joint num diff: {} vs {}'.format(len(jts), self.joint_num)
                jts_3d[:, 0:2] = jts[:, 0:2]
                jts_3d_vis[:, 0] = jts_vis[:]
                jts_3d_vis[:, 1] = jts_vis[:]

            img_path = os.path.join(self.dataset_path, '', 'images', a['image'])
            gt_db.append({
                'image': img_path,
                'center_x': c_x,
                'center_y': c_y,
                'width': width,
                'height': height,
                'flip_pairs': self.flip_pairs,
                'parent_ids': self.parent_ids,
                'joints_3d': jts_3d,
                'joints_3d_vis': jts_3d_vis,
            })

            DEBUG = False
            if DEBUG:
                box = [c_x, c_y, width, height]
                pose = [jts_3d, jts_3d_vis]
                debug_vis(img_path, box, pose)

        with open(cache_file, 'wb') as fid:
            pk.dump(gt_db, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples ared wrote {}'.format(len(gt_db), cache_file))

        return gt_db

    def jnt_bbox_db(self):

        cache_file = os.path.join(self.cache_path, self.name + '_keypoint_jntBBox_db.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))
            return db

        # create train/val split
        with open(os.path.join(self.dataset_path, 'annot', self.image_set_name + '.json')) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            # joints and vis
            jts_3d = np.zeros((self.joint_num, 3), dtype=np.float)
            jts_3d_vis = np.zeros((self.joint_num, 3), dtype=np.float)
            if self.image_set_name != 'test':
                jts = np.array(a['joints'])
                jts[:, 0:2] = jts[:, 0:2] - 1
                jts_vis = np.array(a['joints_vis'])
                assert len(jts) == self.joint_num, 'joint num diff: {} vs {}'.format(len(jts), self.joint_num)
                jts_3d[:, 0:2] = jts[:, 0:2]
                jts_3d_vis[:, 0] = jts_vis[:]
                jts_3d_vis[:, 1] = jts_vis[:]

            if np.sum(jts_3d_vis[:, 0]) < 2:  # only one joint visible, skip
                continue

            u, d, l, r = calc_kpt_bound(jts_3d, jts_3d_vis)
            center = np.array([(l + r) * 0.5, (u + d) * 0.5], dtype=np.float32)
            c_x = center[0]
            c_y = center[1]
            assert c_x >= 1

            w = r - l
            h = d - u

            assert w > 0
            assert h > 0

            if w > self.aspect_ratio * h:
                h = w * 1.0 / self.aspect_ratio
            elif w < self.aspect_ratio * h:
                w = h * self.aspect_ratio

            width = w * 1.25
            height = h * 1.25

            img_path = os.path.join(self.dataset_path, '', 'images', a['image'])
            gt_db.append({
                'image': img_path,
                'center_x': c_x,
                'center_y': c_y,
                'width': width,
                'height': height,
                'flip_pairs': self.flip_pairs,
                'parent_ids': self.parent_ids,
                'joints_3d': jts_3d,
                'joints_3d_vis': jts_3d_vis,
            })

            DEBUG = False
            if DEBUG:
                box = [c_x, c_y, width, height]
                pose = [jts_3d, jts_3d_vis]
                debug_vis(img_path, box, pose)

        with open(cache_file, 'wb') as fid:
            pk.dump(gt_db, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples ared wrote {}'.format(len(gt_db), cache_file))

        return gt_db

    def evaluate(self, preds, save_path):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0  # 3D vector -> 2D vector

        SC_BIAS = 0.6
        threshold = 0.5

        # load ground truth
        gt_file = os.path.join(self.dataset_path, 'annot', 'gt_{}.mat'.format(self.image_set_name))
        dict = loadmat(gt_file)
        dataset_joints = dict['dataset_joints']
        jnt_missing = dict['jnt_missing']
        pos_gt_src = dict['pos_gt_src']
        headboxes_src = dict['headboxes_src']

        pos_pred_src = transpose(preds, [1, 2, 0])

        # get index
        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing  # get visible joints
        uv_error = pos_pred_src - pos_gt_src  # error between predictions and ground truth
        uv_err = np.linalg.norm(uv_error, axis=1)  # l2-Norm

        # get headsize
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS

        # error / headsize
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)

        # number of CKall@threshold 
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)  # invisible get error:0, so need to multiply jnt_visible
        PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold, jnt_visible)
            pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

        # exclude pelvis&thorax
        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head    :', PCKh[head]),
            ('Shoulder:', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow   :', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist   :', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip     :', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee    :', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle   :', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('PCKh@0.5:', np.sum(PCKh * jnt_ratio)),  # weighted sum
            ('PCKh@0.5:', np.sum(pckAll[50, :] * jnt_ratio)),  # weighted sum
            ('PCKh@0.1:', np.sum(pckAll[10, :] * jnt_ratio))  # weighted sum
        ]

        if save_path:
            pred_file = os.path.join(save_path, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        return name_value
