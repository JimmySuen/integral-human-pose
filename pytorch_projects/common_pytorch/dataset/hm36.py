from __future__ import print_function

import os

import numpy as np
import pickle as pk

from .imdb import IMDB

from common.utility.utils import calc_total_skeleton_length, calc_kpt_bound_pad, \
    compute_similarity_transform, calc_total_skeleton_length_bone
from common.utility.visualization import debug_vis

s_hm36_subject_num = 7
HM_subject_idx = [ 1, 5, 6, 7, 8, 9, 11 ]
HM_subject_idx_inv = [ -1, 0, -1, -1, -1, 1, 2, 3, 4, 5, -1, 6 ]

s_hm36_act_num = 15
HM_act_idx = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ]
HM_act_idx_inv = [ -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ]

s_hm36_subact_num = 2
HM_subact_idx = [ 1, 2 ]
HM_subact_idx_inv = [ -1, 0, 1 ]

s_hm36_camera_num = 4
HM_camera_idx = [ 1, 2, 3, 4 ]
HM_camera_idx_inv = [ -1, 0, 1, 2, 3 ]

# 17 joints of Human3.6M:
# 'root', 'Rleg0', 'Rleg1', 'Rleg2', 'Lleg0', 'Lleg1', 'Lleg2', 'torso', 'neck', 'nose', 'head', 'Larm0', 'Larm1', 'Larm2', 'Rarm0', 'Rarm1', 'Rarm2'
# 'root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'

# 18 joints with Thorax:
# 'root', 'Rleg0', 'Rleg1', 'Rleg2', 'Lleg0', 'Lleg1', 'Lleg2', 'torso', 'neck', 'nose', 'head', 'Larm0', 'Larm1', 'Larm2', 'Rarm0', 'Rarm1', 'Rarm2', 'Thorax'
# 'root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist, 'Thorax''
# 0       1       2        3         4       5        6         7        8       9       10      11           12        13        14           15        16       17
# [ 0,      0,      1,       2,        0,      4,       5,        0,      17,      17,      8,     17,           11,        12,       17,          14,       15,      0]

# 16 joints of MPII
# 0-R_Ankle, 1-R_Knee, 2-R_Hip, 3-L_Hip, 4-L_Knee, 5-L_Ankle, 6-Pelvis, 7-Thorax,
# 8-Neck, 9-Head, 10-R_Wrist, 11-R_Elbow, 12-R_Shoulder, 13-L_Shoulder, 14-L_Elbow, 15-L_Wrist

s_org_36_jt_num = 32
s_36_root_jt_idx = 0
s_36_lsh_jt_idx = 11
s_36_rsh_jt_idx = 14
s_36_jt_num = 18
s_36_flip_pairs = np.array([[1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int)
s_36_parent_ids = np.array([0, 0, 1, 2, 0, 4, 5, 0, 17, 17, 8, 17, 11, 12, 17, 14, 15, 0], dtype=np.int)
s_36_bone_jts = np.array([[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                          [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]])
s_mpii_2_hm36_jt = [6, 2, 1, 0, 3, 4, 5, -1, 8, -1, 9, 13, 14, 15, 12, 11, 10, 7]
s_hm36_2_mpii_jt = [3, 2, 1, 4, 5, 6, 0, 17, 8, 10, 16, 15, 14, 11, 12, 13]

s_coco_2_hm36_jt = [-1, 12, 14, 16, 11, 13, 15, -1, -1, 0, -1, 5, 7, 9, 6, 8, 10, -1]

s_posetrack_2_hm36_jt = [-1, 2, 1, 0, 3, 4, 5, -1, 12, 13, 14, 9, 10, 11, 8, 7, 6, -1]

def from_coco_to_hm36_single(pose, pose_vis):
    res_jts = np.zeros((s_36_jt_num, 3), dtype=np.float)
    res_vis = np.zeros((s_36_jt_num, 3), dtype=np.float)

    for i in range(0, s_36_jt_num):
        id1 = i
        id2 = s_coco_2_hm36_jt[i]
        if id2 >= 0:
            res_jts[id1] = pose[id2].copy()
            res_vis[id1] = pose_vis[id2].copy()

    return res_jts.copy(), res_vis.copy()

def from_coco_to_hm36(db):
    for n_sample in range(0, len(db)):
        res_jts, res_vis = from_coco_to_hm36_single(db[n_sample]['joints_3d'], db[n_sample]['joints_3d_vis'])
        db[n_sample]['joints_3d'] = res_jts
        db[n_sample]['joints_3d_vis'] = res_vis

def from_mpii_to_hm36_single(pose, pose_vis):
    res_jts = np.zeros((s_36_jt_num, 3), dtype=np.float)
    res_vis = np.zeros((s_36_jt_num, 3), dtype=np.float)

    for i in range(0, s_36_jt_num):
        id1 = i
        id2 = s_mpii_2_hm36_jt[i]
        if id2 >= 0:
            res_jts[id1] = pose[id2].copy()
            res_vis[id1] = pose_vis[id2].copy()

    return res_jts.copy(), res_vis.copy()

def from_mpii_to_hm36(db):
    for n_sample in range(0, len(db)):
        res_jts, res_vis = from_mpii_to_hm36_single(db[n_sample]['joints_3d'], db[n_sample]['joints_3d_vis'])
        db[n_sample]['joints_3d'] = res_jts
        db[n_sample]['joints_3d_vis'] = res_vis

def from_posetrack_to_hm36(db):
    for n_sample in range(0, len(db)):
        res_jts = np.zeros((s_36_jt_num, 3), dtype=np.float)
        res_vis = np.zeros((s_36_jt_num, 3), dtype=np.float)

        res_jts_nxt = np.zeros((s_36_jt_num, 3), dtype=np.float)
        res_vis_nxt = np.zeros((s_36_jt_num, 3), dtype=np.float)

        for i in range(0, s_36_jt_num):
            id1 = i
            id2 = s_posetrack_2_hm36_jt[i]
            if id2 >= 0:
                res_jts[id1] = db[n_sample]['joints_3d'][id2].copy()
                res_vis[id1] = db[n_sample]['joints_3d_vis'][id2].copy()

                res_jts_nxt[id1] = db[n_sample]['joints_3d_next'][id2].copy()
                res_vis_nxt[id1] = db[n_sample]['joints_3d_vis_next'][id2].copy()

        res_jts[0] = (res_jts[1] + res_jts[4]) * 0.5
        res_vis[0] = res_vis[1] * res_vis[4]
        res_jts[17] = (res_jts[11] + res_jts[14]) * 0.5
        res_vis[17] = res_vis[11] * res_vis[14]
        res_jts[7] = (res_jts[0] + res_jts[8]) * 0.5
        res_vis[7] = res_vis[0] * res_vis[8]
        db[n_sample]['joints_3d'] = res_jts.copy()
        db[n_sample]['joints_3d_vis'] = res_vis.copy()

        res_jts_nxt[0] = (res_jts_nxt[1] + res_jts_nxt[4]) * 0.5
        res_vis_nxt[0] = res_vis_nxt[1] * res_vis_nxt[4]
        res_jts_nxt[17] = (res_jts_nxt[11] + res_jts_nxt[14]) * 0.5
        res_vis_nxt[17] = res_vis_nxt[11] * res_vis_nxt[14]
        res_jts_nxt[7] = (res_jts_nxt[0] + res_jts_nxt[8]) * 0.5
        res_vis_nxt[7] = res_vis_nxt[0] * res_vis_nxt[8]
        db[n_sample]['joints_3d_next'] = res_jts_nxt.copy()
        db[n_sample]['joints_3d_vis_next'] = res_vis_nxt.copy()

def parsing_hm36_gt_file(gt_file):
    keypoints = []
    with open(gt_file, 'r') as f:
        content = f.read()
        content = content.split('\n')
        image_num = int(float(content[0]))
        img_width = content[1].split(' ')[1]
        img_height = content[1].split(' ')[2]
        rot = content[2].split(' ')[1:10]
        trans = content[3].split(' ')[1:4]
        fl = content[4].split(' ')[1:3]
        c_p = content[5].split(' ')[1:3]
        k_p = content[6].split(' ')[1:4]
        p_p = content[7].split(' ')[1:3]
        jt_list = content[8].split(' ')[1:18]
        for i in range(0, image_num):
            keypoints.append(content[9 + i].split(' ')[1:97])

    keypoints = np.asarray([[float(y) for y in x] for x in keypoints])
    keypoints = keypoints.reshape(keypoints.shape[0], keypoints.shape[1] // 3, 3)
    trans = np.asarray([float(y) for y in trans])
    jt_list = np.asarray([int(y) for y in jt_list])
    keypoints = keypoints[:, jt_list - 1, :]

    # add thorax
    thorax = (keypoints[:, s_36_lsh_jt_idx, :] + keypoints[:, s_36_rsh_jt_idx, :]) * 0.5
    thorax = thorax.reshape((thorax.shape[0], 1, thorax.shape[1]))
    keypoints = np.concatenate((keypoints, thorax), axis=1)

    rot = np.asarray([float(y) for y in rot]).reshape((3,3))
    rot = np.transpose(rot)
    fl = np.asarray([float(y) for y in fl])
    c_p = np.asarray([float(y) for y in c_p])
    img_width = np.asarray(float(img_width))
    img_height = np.asarray(float(img_height))
    return keypoints, trans, jt_list, rot, fl, c_p, img_width, img_height

def CamProj(x, y, z, fx, fy, u, v):
    cam_x = x / z * fx
    cam_x = cam_x + u
    cam_y = y / z * fy
    cam_y = cam_y + v
    return cam_x, cam_y

def CamBackProj(cam_x, cam_y, depth, fx, fy, u, v):
    x = (cam_x - u) / fx * depth
    y = (cam_y - v) / fy * depth
    z = depth
    return x, y, z

def joint_to_bone_mat(parent_ids):
    joint_num = len(parent_ids)
    mat = np.zeros((joint_num, joint_num), dtype=int)
    for i in range(0, joint_num):
        p_i = parent_ids[i]
        if p_i != i:
            mat[i][p_i] = -1
            mat[i][i] = 1
        else:
            mat[i][i] = 1
    return np.transpose(mat)

def joint_to_full_pair_mat(joint_num):
    mat = np.zeros((joint_num * (joint_num - 1) / 2, joint_num), dtype=int)
    idx = 0
    for i in range(0, joint_num):
        for j in range(0, joint_num):
            if j > i:
                mat[idx][i] = 1
                mat[idx][j] = -1
                idx = idx + 1
    return np.transpose(mat)

def convert_joint(jts, vis, mat):
    cvt_jts = np.zeros((mat.shape[1]) * 3, dtype = float)
    cvt_jts[0::3] = np.dot(jts[0::3], mat)
    cvt_jts[1::3] = np.dot(jts[1::3], mat)
    cvt_jts[2::3] = np.dot(jts[2::3], mat)

    vis_mat = mat.copy()
    vis_mat[vis_mat!=0] = 1
    cvt_vis = np.zeros((mat.shape[1]) * 3, dtype = float)

    s = np.sum(vis_mat, axis=0)

    cvt_vis[0::3] = np.dot(vis[0::3], vis_mat) == s
    cvt_vis[1::3] = np.dot(vis[1::3], vis_mat) == s
    cvt_vis[2::3] = np.dot(vis[2::3], vis_mat) == s
    return cvt_jts, cvt_vis

def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t

def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2

def from_worldjt_to_imagejt(n_img, joint_num, rot, keypoints, trans, fl, c_p, rect_3d_width, rect_3d_height):
    # project to image space
    pt_3d = np.zeros((joint_num, 3), dtype=np.float)
    pt_2d = np.zeros((joint_num, 3), dtype=np.float)
    for n_jt in range(0, joint_num):
        pt_3d[n_jt] = np.dot(rot, keypoints[n_img, n_jt] - trans)
        pt_2d[n_jt, 0], pt_2d[n_jt, 1] = CamProj(pt_3d[n_jt, 0], pt_3d[n_jt, 1], pt_3d[n_jt, 2], fl[0], fl[1],
                                                 c_p[0], c_p[1])
        pt_2d[n_jt, 2] = pt_3d[n_jt, 2]

    pelvis3d = pt_3d[s_36_root_jt_idx]
    # build 3D bounding box centered on pelvis, size 2000^2
    rect3d_lt = pelvis3d - [rect_3d_width / 2, rect_3d_height / 2, 0]
    rect3d_rb = pelvis3d + [rect_3d_width / 2, rect_3d_height / 2, 0]
    # back-project 3D BBox to 2D image
    rect2d_l, rect2d_t = CamProj(rect3d_lt[0], rect3d_lt[1], rect3d_lt[2], fl[0], fl[1], c_p[0], c_p[1])
    rect2d_r, rect2d_b = CamProj(rect3d_rb[0], rect3d_rb[1], rect3d_rb[2], fl[0], fl[1], c_p[0], c_p[1])

    # Subtract pelvis depth
    pt_2d[:, 2] = pt_2d[:, 2] - pelvis3d[2]
    pt_2d = pt_2d.reshape((joint_num, 3))
    vis = np.ones((joint_num, 3), dtype=np.float)

    return rect2d_l, rect2d_r, rect2d_t, rect2d_b, pt_2d, pt_3d, vis, pelvis3d

class hm36(IMDB):
    def __init__(self, image_set_name, dataset_path, patch_width, patch_height, rect_3d_width, rect_3d_height):
        super(hm36, self).__init__('HM36', image_set_name, dataset_path, patch_width, patch_height)
        self.joint_num = s_36_jt_num
        self.flip_pairs = s_36_flip_pairs
        self.parent_ids = s_36_parent_ids
        self.idx2name = ['root', 'R-hip', 'R-knee', 'R-ankle', 'L-hip', 'L-knee', 'L-ankle', 'torso', 'neck', 'nose',
                         'head', 'L-shoulder', 'L-elbow', 'L-wrist', 'R-shoulder', 'R-elbow', 'R-wrist', 'thorax']

        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height

        self.aspect_ratio = 1.0 * patch_width / patch_height

        self.mean_bone_length = 0

    def _H36FolderName(self, subject_id, act_id, subact_id, camera_id):
        return "s_%02d_act_%02d_subact_%02d_ca_%02d" % \
               (HM_subject_idx[subject_id], HM_act_idx[act_id], HM_subact_idx[subact_id], HM_camera_idx[camera_id])

    def _H36ImageName(self, folder_name, frame_id):
        return "%s_%06d.jpg" % (folder_name, frame_id + 1)

    def _AllHuman36Folders(self, subject_list_):
        subject_list = subject_list_[:]
        if len(subject_list) == 0:
            for i in range(0, s_hm36_subject_num):
                subject_list.append(i)
        folders = []
        for i in range(0, len(subject_list)):
            for j in range(0, s_hm36_act_num):
                for m in range(0, s_hm36_subact_num):
                    for n in range(0, s_hm36_camera_num):
                        folders.append(self._H36FolderName(subject_list[i], j, m, n))
        return folders

    def _sample_dataset(self, image_set_name):
        if image_set_name == 'train':
            sample_num = 200
            step = -1
            folder_start = 0
            folder_end = 600
            folders = self._AllHuman36Folders([0, 1, 2, 3, 4])
        elif image_set_name == 'trainfull':
            sample_num = -1
            step = 1
            folder_start = 0
            folder_end = 600
            folders = self._AllHuman36Folders([0, 1, 2, 3, 4])
        elif image_set_name == 'trainsample2':
            sample_num = -1
            step = 2
            folder_start = 0
            folder_end = 600
            folders = self._AllHuman36Folders([0, 1, 2, 3, 4])
        elif image_set_name == 'trainsample10':
            sample_num = -1
            step = 10
            folder_start = 0
            folder_end = 600
            folders = self._AllHuman36Folders([0, 1, 2, 3, 4])
        elif image_set_name == 'valid':
            sample_num = 40
            step = -1
            folder_start = 0
            folder_end = 240
            folders = self._AllHuman36Folders([5, 6])
        elif image_set_name == 'validmin':
            sample_num = 10
            step = -1
            folder_start = 0
            folder_end = 240
            folders = self._AllHuman36Folders([5, 6])
        elif image_set_name == 'validfull':
            sample_num = -1
            step = 1
            folder_start = 0
            folder_end = 240
            folders = self._AllHuman36Folders([5, 6])
        elif image_set_name == 'validsample2':
            sample_num = -1
            step = 2
            folder_start = 0
            folder_end = 240
            folders = self._AllHuman36Folders([5, 6])
        elif image_set_name == 'validsample10':
            sample_num = -1
            step = 10
            folder_start = 0
            folder_end = 240
            folders = self._AllHuman36Folders([5, 6])
        elif 'act' in image_set_name:
            act_id = int(image_set_name[-2:])
            sample_num = 40
            step = -1
            folder_start = 0
            folders = []
            s_list = [5, 6]
            for i in range(0, len(s_list)):
                for m in range(0, s_hm36_subact_num):
                    for n in range(0, s_hm36_camera_num):
                        folders.append(self._H36FolderName(s_list[i], act_id, m, n))
            folder_end = len(folders)
        else:
            print("Error!!!!!!!!! Unknown hm36 sub set!")
            assert 0
        return folders, sample_num, step, folder_start, folder_end

    def gt_db(self):
        folders, sample_num, step, folder_start, folder_end = self._sample_dataset(self.image_set_name)

        db = None
        cache_file = os.path.join(self.cache_path, self.name + '_keypoint_db_sample' + str(sample_num) + '.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))

        if db != None:
            self.num_sample_single = len(db)
            return db

        gt_db = []
        for n_folder in range(folder_start, folder_end):
            print('Loading folder ', n_folder, ' in ', len(folders))

            # load ground truth
            keypoints, trans, jt_list, rot, fl, c_p, img_width, img_height = parsing_hm36_gt_file(
                os.path.join(self.dataset_path, "annot", folders[n_folder], 'matlab_meta.txt'))

            # random sample redundant video sequence
            if sample_num > 0:
                img_index = np.random.choice(keypoints.shape[0], sample_num, replace=False)
            else:
                img_index = np.arange(keypoints.shape[0])
                img_index = img_index[0:keypoints.shape[0]:step]

            for n_img_ in range(0, img_index.shape[0]):
                n_img = img_index[n_img_]
                image_name = os.path.join(folders[n_folder], self._H36ImageName(folders[n_folder], n_img))
                assert keypoints.shape[1] == self.joint_num

                rect2d_l, rect2d_r, rect2d_t, rect2d_b, pt_2d, pt_3d, vis, pelvis3d = \
                    from_worldjt_to_imagejt(n_img, self.joint_num, rot, keypoints, trans, fl, c_p, self.rect_3d_width, self.rect_3d_height)

                skeleton_length = calc_total_skeleton_length_bone(pt_3d, s_36_bone_jts)

                gt_db.append({
                    'image': os.path.join(self.dataset_path, '', 'images', image_name),
                    'center_x': (rect2d_l + rect2d_r) * 0.5,
                    'center_y': (rect2d_t + rect2d_b) * 0.5,
                    'width': (rect2d_r - rect2d_l),
                    'height': (rect2d_b - rect2d_t),
                    'flip_pairs': self.flip_pairs,
                    'parent_ids': self.parent_ids,
                    'joints_3d': pt_2d, # [org_img_x, org_img_y, depth - root_depth]
                    'joints_3d_vis': vis,

                    'joints_3d_cam': pt_3d, # [X, Y, Z] in camera coordinate
                    'pelvis': pelvis3d,
                    'fl': fl,
                    'c_p': c_p,

                    'bone_len': skeleton_length
                })


        with open(cache_file, 'wb') as fid:
            pk.dump(gt_db, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples ared wrote {}'.format(len(gt_db), cache_file))

        self.num_sample_single = len(gt_db)

        return gt_db

    def dt_db(self, det_bbox_src):
        print("Using Detector:", det_bbox_src)

        self.detector = det_bbox_src
        folders, sample_num, step, folder_start, folder_end = self._sample_dataset(self.image_set_name)

        dt_cache_file = os.path.join(self.cache_path, self.name + '_keypoint_dt_db_sample' + str(sample_num) + '.pkl')
        if os.path.exists(dt_cache_file):
            with open(dt_cache_file, 'rb') as fid:
                dt_db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, dt_cache_file, len(dt_db)))
            return dt_db

        gt_cache_file = os.path.join(self.cache_path, self.name + '_keypoint_db_sample' + str(sample_num) + '.pkl')

        if os.path.exists(gt_cache_file):
            with open(gt_cache_file, 'rb') as fid:
                gt_db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, gt_cache_file, len(gt_db)))
        else:
            assert 0, gt_cache_file + ' not exist...'

        self.num_sample_single = len(gt_db)
        self.mean_bone_length = np.asarray([item['bone_len'] for item in gt_db]).mean()

        # update bbox using detection result
        print("Updating BBox from detector")
        bbox_file = os.path.join(self.cache_path, 'detection', det_bbox_src, 'kpts_bbox.pkl')
        with open(bbox_file, 'rb') as fid:
            bbox_list = pk.load(fid)

        assert len(bbox_list) == len(gt_db)
        for idx in range(len(gt_db)):
            box = bbox_list[idx]
            center_x = (box[0] + box[2]) * 0.5
            center_y = (box[1] + box[3]) * 0.5

            width = box[2] - box[0]
            height = box[3] - box[1]

            if width > self.aspect_ratio * height:
                height = width * 1.0 / self.aspect_ratio
            elif width < self.aspect_ratio * height:
                width = height * self.aspect_ratio

            width = width * 1.25
            height = height * 1.25

            gt_db[idx]['center_x'] = center_x
            gt_db[idx]['center_y'] = center_y
            gt_db[idx]['width']    = width
            gt_db[idx]['height']   = height

            DEBUG = False
            if DEBUG:
                box = [center_x, center_y, width, height]
                pose = []
                debug_vis(os.path.join(gt_db[idx]['image']), box, pose)

        self.num_sample_single = len(gt_db)

        return gt_db

    def jnt_bbox_db(self):
        db = None
        folders, sample_num, step, folder_start, folder_end = self._sample_dataset(self.image_set_name)
        cache_file = os.path.join(self.cache_path, self.name + '_keypoint_jntBBox_db_sample' + str(sample_num) + '.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))

        if db != None:
            self.num_sample_single = len(db)
            self.mean_bone_length = np.asarray([item['bone_len'] for item in db]).mean()
            return db

        jnt_bbox_db = []
        for n_folder in range(folder_start, folder_end):
            print('Loading folder ', n_folder, ' in ', len(folders))

            # load ground truth
            keypoints, trans, jt_list, rot, fl, c_p, img_width, img_height = parsing_hm36_gt_file(
                os.path.join(self.dataset_path, "annot", folders[n_folder], 'matlab_meta.txt'))

            # random sample redundant video sequence
            if sample_num > 0:
                img_index = np.random.choice(keypoints.shape[0], sample_num, replace=False)
            else:
                img_index = np.arange(keypoints.shape[0])
                img_index = img_index[0:keypoints.shape[0]:step]

            for n_img_ in range(0, img_index.shape[0]):
                n_img = img_index[n_img_]
                image_name = os.path.join(folders[n_folder], self._H36ImageName(folders[n_folder], n_img))
                assert keypoints.shape[1] == self.joint_num

                _, _, _, _, pt_2d, pt_3d, vis, pelvis3d = \
                    from_worldjt_to_imagejt(n_img, self.joint_num, rot, keypoints, trans, fl, c_p, self.rect_3d_width,
                                             self.rect_3d_height)

                c_x, c_y, w, h = calc_kpt_bound_pad(pt_2d, vis, self.aspect_ratio)

                pt_3d_relative = pt_3d - pt_3d[0]
                skeleton_length = calc_total_skeleton_length(pt_3d_relative, s_36_parent_ids)

                jnt_bbox_db.append({
                    'image': os.path.join(self.dataset_path, '', 'images', image_name),
                    'center_x': c_x,
                    'center_y': c_y,
                    'width': w,
                    'height': h,
                    'flip_pairs': self.flip_pairs,
                    'parent_ids': self.parent_ids,
                    'joints_3d': pt_2d,  # [org_img_x, org_img_y, depth - root_depth]
                    'joints_3d_vis': vis,

                    'joints_3d_cam': pt_3d,  # [X, Y, Z] in camera coordinate
                    'pelvis': pelvis3d,
                    'fl': fl,
                    'c_p': c_p,

                    'joints_3d_relative': pt_3d_relative,  # [X-root, Y-root, Z-root] in camera coordinate
                    'bone_len': skeleton_length
                })

        self.mean_bone_length = np.asarray([item['bone_len'] for item in jnt_bbox_db]).mean()


        with open(cache_file, 'wb') as fid:
            pk.dump(jnt_bbox_db, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples ared wrote {}'.format(len(jnt_bbox_db), cache_file))

        self.num_sample_single = len(jnt_bbox_db)

        return jnt_bbox_db

    def get_mean_bone_length(self):
        return self.mean_bone_length

    def evaluate(self, preds, save_path):
        preds = preds[:, :, 0:3]

        gts = self.gt_db()

        sample_num = preds.shape[0]
        joint_num = self.joint_num
        # flip_pair = self.flip_pairs
        parent_ids = self.parent_ids

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

        metrics_num = 3
        t_start = 5
        t_step = 100
        t_end = 205
        coord_num = int(4 + (t_end - t_start) / t_step) # 4 for all, x, y, z
        avg_errors = [] # [metrics_num, len(eval_jt), coord_num]
        for n_m in range(0, metrics_num):
            avg_errors.append(np.zeros((len(eval_jt), coord_num), dtype=np.float))

        pred_to_save = []
        for n_sample in range(0, sample_num):
            gt = gts[n_sample]
            # Org image info
            fl = gt['fl'][0:2]
            c_p = gt['c_p'][0:2]

            gt_3d_root = np.reshape(gt['pelvis'], (1, 3))
            gt_2d_kpt = gt['joints_3d'].copy()
            gt_vis = gt['joints_3d_vis'].copy()

            # get camera depth from root joint
            pre_2d_kpt = preds[n_sample].copy()
            pre_2d_kpt[:, 2] = pre_2d_kpt[:, 2] + gt_3d_root[0, 2]
            gt_2d_kpt[:, 2] = gt_2d_kpt[:, 2] + gt_3d_root[0, 2]

            # back project
            pre_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)
            gt_3d_kpt = np.zeros((joint_num, 3), dtype=np.float)
            for n_jt in range(0, joint_num):
                pre_3d_kpt[n_jt, 0], pre_3d_kpt[n_jt, 1], pre_3d_kpt[n_jt, 2] = \
                    CamBackProj(pre_2d_kpt[n_jt, 0], pre_2d_kpt[n_jt, 1], pre_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])
                gt_3d_kpt[n_jt, 0], gt_3d_kpt[n_jt, 1], gt_3d_kpt[n_jt, 2] = \
                    CamBackProj(gt_2d_kpt[n_jt, 0], gt_2d_kpt[n_jt, 1], gt_2d_kpt[n_jt, 2], fl[0], fl[1], c_p[0],
                                c_p[1])

            # bone
            j2b_mat = joint_to_bone_mat(parent_ids)
            pre_3d_bone, bone_vis = convert_joint(np.reshape(pre_3d_kpt, joint_num * 3),
                                                  np.reshape(gt_vis, joint_num * 3), j2b_mat)
            gt_3d_bone, bone_vis = convert_joint(np.reshape(gt_3d_kpt, joint_num * 3),
                                                 np.reshape(gt_vis, joint_num * 3), j2b_mat)
            pre_3d_bone = np.reshape(pre_3d_bone, (joint_num, 3))
            gt_3d_bone = np.reshape(gt_3d_bone, (joint_num, 3))

            # align
            pre_3d_kpt_align = rigid_align(pre_3d_kpt, gt_3d_kpt)

            diffs = [] # [metrics_num, joint_num * 3]

            # should align root, required by protocol #1
            pre_3d_kpt = pre_3d_kpt - pre_3d_kpt [0]
            gt_3d_kpt  = gt_3d_kpt - gt_3d_kpt [0]
            pre_3d_kpt_align = pre_3d_kpt_align - pre_3d_kpt_align [0]

            diffs.append((pre_3d_kpt - gt_3d_kpt))  # Avg joint error
            diffs.append((pre_3d_bone - gt_3d_bone))
            diffs.append((pre_3d_kpt_align - gt_3d_kpt))

            pred_to_save.append({'pred': pre_3d_kpt,
                                 'align_pred': pre_3d_kpt_align,
                                 'gt': gt_3d_kpt})

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

        name_value = [
            ('hm36_root     :', avg_errors[0][0][0]),
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
            ('hm36_17j_align:', avg_errors[2][11][0]),
            ('hm36_17j_x    :', avg_errors[0][11][1]),
            ('hm36_17j_y    :', avg_errors[0][11][2]),
            ('hm36_17j_z    :', avg_errors[0][11][3]),
        ]

        return name_value
