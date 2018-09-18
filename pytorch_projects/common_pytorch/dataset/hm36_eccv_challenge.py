from __future__ import print_function
import os
import cv2
import glob
import json
import numpy as np
from numpy import genfromtxt
import pickle as pk

from .imdb import IMDB
from common.utility.utils import calc_total_skeleton_length, compute_similarity_transform, calc_kpt_bound
from common.utility.visualization import debug_vis, cv_draw_joints

from .hm36 import s_org_36_jt_num, s_36_root_jt_idx, s_36_lsh_jt_idx, s_36_rsh_jt_idx\
    , s_36_jt_num, s_36_flip_pairs, s_36_parent_ids\
    , s_mpii_2_hm36_jt, s_hm36_2_mpii_jt, s_posetrack_2_hm36_jt\
    , rigid_align, from_mpii_to_hm36_single


def jnt_bbox_db_core(name, cache_path, dataset_path, image_set_name, image_names, joint_num, aspect_ratio
                     , flip_pairs, parent_ids):
    '''
    This function is to 1)get aligned 2d pose;  2)record align params;
    3)generate bbox around aligned 2d pose; 4)calc skeleton length
    It's the very db used to train and val
    :return:
    '''
    cache_file = '{}_keypoint_jnt_bbox_db.pkl'.format(name)
    cache_file = os.path.join(cache_path, cache_file)
    db = None

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            db = pk.load(fid)
        print('{} gt db loaded from {}, {} samples are loaded'.format(name, cache_file, len(db)))

    if db != None:
        return db

    img_pred_file = '{}_{}samples_img_res.pkl'.format('HM36_eccv_challenge_' + image_set_name, len(image_names))
    img_pred_file = os.path.join(dataset_path, image_set_name, img_pred_file)
    with open(img_pred_file, 'rb') as fid:
        img_pred = pk.load(fid)

    dt_db = []
    for idx in range(len(image_names)):
        img_path = os.path.join(dataset_path, image_set_name, 'IMG', '%05d.jpg' % (idx + 1))
        pred_pose_in_img_wz_score = img_pred[idx]["kpts"]  # 18x3, already in hm36 skeleton structure
        pred_pose_vis = img_pred[idx]["vis"]

        if image_set_name == 'Test':
            # only thing need to do: generate bbox around kpts
            mask = np.where(pred_pose_vis[:, 0] > 0)  # only align visible joints
            u, d, l, r = calc_kpt_bound(pred_pose_in_img_wz_score[mask[0], 0:2], pred_pose_vis[mask[0], 0:2])
            align_joints_2d_wz = joints_2d_vis = gtPose = np.zeros((18, 3))
            skeleton_length = s = rot = t = 0
        elif image_set_name in ['Train', 'Val']:
            # process pose
            gt_file = os.path.join(dataset_path, image_set_name, 'POSE', '%05d.csv' % (idx + 1))
            gtPose = genfromtxt(gt_file, delimiter=',')

            # add thorax
            if joint_num == s_36_jt_num:
                thorax = (gtPose[s_36_lsh_jt_idx] + gtPose[s_36_rsh_jt_idx]) * 0.5
                thorax = thorax.reshape((1, 3))
                gtPose = np.concatenate((gtPose, thorax), axis=0)
            assert len(gtPose) == s_36_jt_num, "#Joint Must be 18, Now #Joint %d" % len(gtPose)
            # align
            mask = np.where(pred_pose_vis[:, 0] > 0)  # only align visible joints
            target_pose = pred_pose_in_img_wz_score[mask[0], 0:2]
            from_pose = gtPose[mask[0], 0:2]
            _, Z, rot, s, t = compute_similarity_transform(target_pose, from_pose, compute_optimal_scale=True)


            align_joints_2d_wz = s * gtPose[:, 0:2].dot(rot) + t
            align_joints_2d_wz = np.concatenate((align_joints_2d_wz, gtPose[:, 2:3] * s), axis=1)

            joints_2d_vis = np.ones(align_joints_2d_wz.shape, dtype=np.float)

            # other
            skeleton_length = calc_total_skeleton_length(gtPose, s_36_parent_ids)

            # generate bbox
            u, d, l, r = calc_kpt_bound(align_joints_2d_wz, joints_2d_vis)


        center_x = (l + r) * 0.5
        center_y = (u + d) * 0.5
        assert center_x >= 1

        w = r - l
        h = d - u
        assert w > 0
        assert h > 0

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        w *= 1.25
        h *= 1.25

        dt_db.append({
            'image': img_path,
            'flip_pairs': flip_pairs,
            'parent_ids': parent_ids,

            # pose
            'joints_3d': align_joints_2d_wz,  # [org_img_x, org_img_y, depth - root_depth]
            'joints_3d_vis': joints_2d_vis,
            'joints_3d_relative': gtPose,  # [X-root, Y-root, Z-root] in camera coordinate, substracted by root
            'bone_len': skeleton_length,

            # bbox
            'center_x': center_x,
            'center_y': center_y,
            'width': w,
            'height': h,

            # align
            's': s,
            'rot': rot,
            't': t
        })

        DEBUG = False
        if DEBUG:
            bbox = [center_x, center_y, w, h]
            pose = [align_joints_2d_wz, joints_2d_vis]
            debug_vis(img_path, bbox, pose)


    with open(cache_file, 'wb') as fid:
        pk.dump(dt_db, fid, pk.HIGHEST_PROTOCOL)
    print('{} samples ared wrote {}'.format(len(dt_db), cache_file))

    return dt_db


class hm36_eccv_challenge(IMDB):

    def __init__(self, image_set_name, dataset_path, patch_width, patch_height, rect_3d_width, rect_3d_height):
        super(hm36_eccv_challenge, self).__init__('HM36_eccv_challenge', image_set_name, dataset_path, patch_width,
                                                  patch_height)
        self.joint_num = s_36_jt_num
        self.flip_pairs = s_36_flip_pairs
        self.parent_ids = s_36_parent_ids
        self.idx2name = ['root', 'R-hip', 'R-knee', 'R-ankle', 'L-hip', 'L-knee', 'L-ankle', 'torso', 'neck', 'nose',
                         'head', 'L-shoulder', 'L-elbow', 'L-wrist', 'R-shoulder', 'R-elbow', 'R-wrist', 'thorax']

        self.aspect_ratio = 1.0 * patch_width / patch_height
        self.mean_bone_length = 0  #4465.869 for Val, 4522.828 for Train, 4502.881 for average

        if image_set_name == "Train" or image_set_name == "Val" or image_set_name == "Test":
            self.image_names = sorted(glob.glob(os.path.join(self.dataset_path, image_set_name, "IMG") + "/*.jpg"))
        elif image_set_name == "TrainVal":
            self.image_names = []
            self.image_names.append(sorted(glob.glob(os.path.join(self.dataset_path, "Train", "IMG") + "/*.jpg")))
            self.image_names.append(sorted(glob.glob(os.path.join(self.dataset_path, "Val", "IMG") + "/*.jpg")))
        else:
            assert 0
        self.detector = ''

    def org_db(self):
        cache_file = '{}_keypoint_org_box_db.pkl'.format(self.name)
        cache_file = os.path.join(self.cache_path, cache_file)
        db = None

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))

        if db != None:
            return db

        dt_db = []
        for idx in range(len(self.image_names)):
            img_path = os.path.join(self.dataset_path, self.image_set_name, 'IMG', '%05d.jpg' % (idx + 1))

            if self.image_set_name == 'Test':
                assert 0
            elif self.image_set_name in ['Train', 'Val']:
                # process pose
                gt_file = os.path.join(self.dataset_path, self.image_set_name, 'POSE', '%05d.csv' % (idx + 1))
                gtPose = genfromtxt(gt_file, delimiter=',')

                # add thorax
                if self.joint_num == s_36_jt_num:
                    thorax = (gtPose[s_36_lsh_jt_idx] + gtPose[s_36_rsh_jt_idx]) * 0.5
                    thorax = thorax.reshape((1, 3))
                    gtPose = np.concatenate((gtPose, thorax), axis=0)
                assert len(gtPose) == s_36_jt_num, "#Joint Must be 18, Now #Joint %d" % len(gtPose)

                joints_2d_vis = np.ones(gtPose.shape, dtype=np.float)

            center_x = 500
            center_y = 500

            w = 1
            h = 1000

            if w > self.aspect_ratio * h:
                h = w * 1.0 / self.aspect_ratio
            elif w < self.aspect_ratio * h:
                w = h * self.aspect_ratio

            dt_db.append({
                'image': img_path,
                'flip_pairs': self.flip_pairs,
                'parent_ids': self.parent_ids,

                # pose
                'joints_3d': gtPose,  # [org_img_x, org_img_y, depth - root_depth]
                'joints_3d_vis': joints_2d_vis,
                'joints_3d_relative': gtPose,  # [X-root, Y-root, Z-root] in camera coordinate, substracted by root

                # bbox
                'center_x': center_x,
                'center_y': center_y,
                'width': w,
                'height': h,
            })

        with open(cache_file, 'wb') as fid:
            pk.dump(dt_db, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples ared wrote {}'.format(len(dt_db), cache_file))

        return dt_db

    def jnt_bbox_db(self):
        if self.image_set_name == "Train" or self.image_set_name == "Val" or self.image_set_name == "Test":
            db = jnt_bbox_db_core(self.name, self.cache_path, self.dataset_path, self.image_set_name,
                             self.image_names, self.joint_num, self.aspect_ratio, self.flip_pairs, self.parent_ids)
            self.num_sample_single = len(db)
            self.mean_bone_length = np.asarray([item['bone_len'] for item in db]).mean()
            return db
        elif self.image_set_name == "TrainVal":
            db_t = jnt_bbox_db_core(self.name.replace('Val', ''), self.cache_path.replace('Val', ''),
                                    self.dataset_path, self.image_set_name.replace('Val', ''), self.image_names[0],
                                    self.joint_num, self.aspect_ratio, self.flip_pairs, self.parent_ids)
            db_v = jnt_bbox_db_core(self.name.replace('Train', ''), self.cache_path.replace('Train', ''),
                                    self.dataset_path, self.image_set_name.replace('Train', ''), self.image_names[1],
                                    self.joint_num, self.aspect_ratio, self.flip_pairs, self.parent_ids)
            db = db_t + db_v
            self.num_sample_single = len(db)
            self.mean_bone_length = np.asarray([item['bone_len'] for item in db]).mean()
            return db
        else:
            assert 0

    def dt_db(self, det_bbox_src):
        '''
        This function is to organize image path, maskRCNN bbox, etc into data structure,
        for the purpose of predicting 2d pose by a mpii pose estimator.
        So only image&bbox related are useful, others by default set to zero
        :param det_bbox_src:
        :return:
        '''
        self.detector = det_bbox_src
        cache_file = '{}_bbox_dt_{}_db.pkl'.format(self.name, self.detector)
        cache_file = os.path.join(self.cache_path, cache_file)
        db = None

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))

        if db != None:
            self.num_sample_single = len(db)
            return db

        dt_db = []
        for idx in range(len(self.image_names)):
            img_path = os.path.join(self.dataset_path, self.image_set_name, 'IMG', '%05d.jpg' % (idx + 1))
            bbox_file = os.path.join(self.dataset_path, self.image_set_name,
                                     'detection', self.detector, '%05d.pkl' % (idx + 1))

            # process bbox
            with open(bbox_file, 'rb') as fid:
                bbox = pk.load(fid)

            assert len(bbox) == 1, "Cannot be %d bbox for image %s"%(len(bbox), img_path)
            box = bbox[0]
            center_x = (box[0] + box[2]) * 0.5
            center_y = (box[1] + box[3]) * 0.5
            score = box[4]

            width  = box[2] - box[0]
            height = box[3] - box[1]

            if width > self.aspect_ratio * height:
                height = width * 1.0 / self.aspect_ratio
            elif width < self.aspect_ratio * height:
                width = height * self.aspect_ratio

            width  = width * 1.1
            height = height * 1.1

            dt_db.append({
                'image': img_path,
                'flip_pairs': self.flip_pairs,
                'parent_ids': self.parent_ids,

                # joint, useless
                'joints_3d': np.zeros((18,3)),   # [org_img_x, org_img_y, depth - root_depth]
                'joints_3d_vis': np.zeros((18,3)),
                'joints_3d_relative': np.zeros((18,3)), # [X-root, Y-root, Z-root] in camera coordinate, substracted by root
                'bone_len':0,

                # bbox
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'score': score,
            })

            DEBUG = False
            if DEBUG:
                bbox = [center_x, center_y, width, height]
                pose = []
                debug_vis(img_path, bbox, pose)

        with open(cache_file, 'wb') as fid:
            pk.dump(dt_db, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples ared wrote {}'.format(len(dt_db), cache_file))

        self.num_sample_single = len(dt_db)
        return dt_db

    def get_mean_bone_length(self):
        return self.mean_bone_length

    def evaluate(self, preds, save_path, is_save=False):

        jnt_bbox_db = self.jnt_bbox_db()

        preds = preds[:, :, 0:3]

        sample_num = preds.shape[0]
        assert len(jnt_bbox_db) == sample_num, "#db samples  %d != #preds %d"%(len(jnt_bbox_db), sample_num)

        joint_num = self.joint_num
        # flip_pair = self.flip_pairs
        # parent_ids = self.parent_ids

        # 18 joints:
        # 'root', 'Rleg0', 'Rleg1', 'Rleg2', 'Lleg0', 'Lleg1', 'Lleg2', 'torso', 'neck', 'nose', 'head', 'Larm0', 'Larm1', 'Larm2', 'Rarm0', 'Rarm1', 'Rarm2', 'Thorax'
        # 'root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist, 'Thorax''
        # 0       1       2        3         4       5        6         7        8       9       10      11           12        13        14           15        16       17
        # joint_names = ['root', 'Hip', 'Knee', 'Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'Shoulder', 'Elbow', 'Wrist', '17j', '16j', '14j', '13j']
        eval_jt = [[0], [1, 4], [2, 5], [3, 6], [7], [8], [9], [10], [11, 14], [12, 15], [13, 16],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                   [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17],
                   [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16],
                   [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]]

        metrics_num = 2
        t_start = 5
        # t_step = 5
        t_step = 100
        t_end = 205
        coord_num = int(4 + (t_end - t_start) / t_step) # 4 for all, x, y, z
        avg_errors = [] # [metrics_num, len(eval_jt), coord_num]
        for n_m in range(0, metrics_num):
            avg_errors.append(np.zeros((len(eval_jt), coord_num), dtype=np.float))

        if is_save:
            p_save_path = os.path.join(save_path, self.image_set_name + '_PRED')
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            print("save prediction CSV into", p_save_path)

        test_store = []
        for n_sample in range(0, sample_num):
            sample = jnt_bbox_db[n_sample]

            pred_3d_kpt = preds[n_sample].copy()
            assert len(pred_3d_kpt) == 18
            gt_3d_kpt  = sample['joints_3d_relative'][:]

            if is_save:
                with open(os.path.join(p_save_path, '%05d.csv' % (n_sample + 1)),'w') as fid:
                    for n_j in range(17):  #only save 17 joints, subtracted by root
                        wstring = '{:.2f},{:.2f},{:.2f}'.format(
                            pred_3d_kpt[n_j][0], pred_3d_kpt[n_j][1], pred_3d_kpt[n_j][2])
                        print(wstring, file=fid)

            if self.image_set_name == 'Test':
                item = dict()
                item['image_id'] = n_sample + 1
                item['keypoints'] = []
                kpts = list()
                for n_j in range(len(pred_3d_kpt) - 1):
                    for j in range(3):
                        kpts.append(pred_3d_kpt[n_j][j])
                item['keypoints'].append(kpts)
                test_store.append(item)
                continue

            # align
            pred_3d_kpt_align = rigid_align(pred_3d_kpt, gt_3d_kpt)

            diffs = [] # [metrics_num, joint_num * 3]
            diffs.append((pred_3d_kpt - gt_3d_kpt))  # Avg joint error
            diffs.append((pred_3d_kpt_align - gt_3d_kpt))

            for n_m in range(0, metrics_num):
                e_jt = []
                e_jt_x = []
                e_jt_y = []
                e_jt_z = []
                e_jt_pck = [[] for i in range(t_start, t_end, t_step)]
                for n_jt in range(0, joint_num):
                    t_dis = np.linalg.norm(diffs[n_m][n_jt])
                    e_jt.append(t_dis)
                    e_jt_x.append(abs(diffs[n_m][n_jt][0]))
                    e_jt_y.append(abs(diffs[n_m][n_jt][1]))
                    e_jt_z.append(abs(diffs[n_m][n_jt][2]))
                    for i in range(t_start, t_end, t_step):
                        e_jt_pck[int((i - t_start) / t_step)].append(int(t_dis < i))

                for n_eval_jt in range(0, len(eval_jt)):
                    e = 0
                    e_x = 0
                    e_y = 0
                    e_z = 0
                    e_pck = [0 for i in range(t_start, t_end, t_step)]
                    for n_jt in range(0, len(eval_jt[n_eval_jt])):
                        e = e + e_jt[eval_jt[n_eval_jt][n_jt]]
                        e_x = e_x + e_jt_x[eval_jt[n_eval_jt][n_jt]]
                        e_y = e_y + e_jt_y[eval_jt[n_eval_jt][n_jt]]
                        e_z = e_z + e_jt_z[eval_jt[n_eval_jt][n_jt]]
                        for i in range(t_start, t_end, t_step):
                            e_pck[int((i - t_start) / t_step)] = \
                                e_pck[int((i - t_start) / t_step)] + \
                                e_jt_pck[int((i - t_start) / t_step)][eval_jt[n_eval_jt][n_jt]]

                    avg_errors[n_m][n_eval_jt][0] = avg_errors[n_m][n_eval_jt][0] + e / float(len(eval_jt[n_eval_jt]))
                    avg_errors[n_m][n_eval_jt][1] = avg_errors[n_m][n_eval_jt][1] + e_x / float(len(eval_jt[n_eval_jt]))
                    avg_errors[n_m][n_eval_jt][2] = avg_errors[n_m][n_eval_jt][2] + e_y / float(len(eval_jt[n_eval_jt]))
                    avg_errors[n_m][n_eval_jt][3] = avg_errors[n_m][n_eval_jt][3] + e_z / float(len(eval_jt[n_eval_jt]))
                    for i in range(t_start, t_end, t_step):
                        avg_errors[n_m][n_eval_jt][4 + int((i - t_start) / t_step)] = \
                            avg_errors[n_m][n_eval_jt][4 + int((i - t_start) / t_step)] + \
                            e_pck[int((i - t_start) / t_step)] / float(len(eval_jt[n_eval_jt]))

        for n_m in range(0, metrics_num):
            avg_errors[n_m] = avg_errors[n_m] / sample_num

        if self.image_set_name == 'Test':
            with open(os.path.join(save_path, self.image_set_name + '_Prediction.JSON'), 'w') as fid:
                json.dump(test_store, fid)

        name_value = [
            ('hm36_root     ;', avg_errors[0][0][0]),
            ('hm36_Hip      :', avg_errors[0][1][0]),
            ('hm36_Knee     :', avg_errors[0][2][0]),
            ('hm36_Ankle    :', avg_errors[0][3][0]),
            ('hm36_Torso    :', avg_errors[0][4][0]),
            ('hm36_Neck     :', avg_errors[0][5][0]),
            ('hm36_Nose     :', avg_errors[0][6][0]),
            ('hm36_Head     :', avg_errors[0][7][0]),
            ('hm36_Shoulder :', avg_errors[0][8][0]),
            ('hm36_Elbow    :', avg_errors[0][9][0]),
            ('hm36_Wrist    :', avg_errors[0][10][0]),
            ('hm36_17j      :', avg_errors[0][11][0]),
            ('hm36_17j_align:', avg_errors[1][11][0]),
            ('hm36_17j_x    :', avg_errors[0][11][1]),
            ('hm36_17j_y    :', avg_errors[0][11][2]),
            ('hm36_17j_z    :', avg_errors[0][11][3]),
        ]

        return name_value