from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import json_tricks as json
from collections import defaultdict

import numpy as np
from common.nms.nms import oks_nms

# coco api
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .imdb import IMDB

# 17 joints of COCO:
# 0 - nose,  1 - left_eye,  2 - right_eye, 3 - left_ear, 4 - right_ear
# 5 - left_shoulder, 6 - right_shoulder, 7 - left_elbow, 8 - right_elbow, 9 - left_wrist, 10 - right_wrist
# 11 - left_hip, 12 - right_hip, 13 - left_knee, 14 - right_knee. 15 - left_ankle, 16 - right_ankle

class coco(IMDB):
    def __init__(self, image_set_name, dataset_path, patch_width, patch_height, *args):
        """
        fill basic information to initialize imdb
        :param image_set_name: train2014, val2014, test2015
        :param root_path: 'data', will write 'rpn_data', 'cache'
        :param data_path: 'data/coco'
        """
        super(coco, self).__init__('COCO',
                                   image_set_name,
                                   dataset_path,
                                   patch_width,
                                   patch_height)
        self.image_thre = 0.0
        self.oks_thre = 0.9
        self.in_vis_thre = 0.2
        self.all_boxes = None

        self.aspect_ratio = self.patch_width * 1.0 / self.patch_height
        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        print('classes', self.classes)
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        print('num_images', self.num_images)
        self.data_name = image_set_name

        self.joint_num = 17
        self.flip_pairs = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]],
                                   dtype=np.int)
        self.parent_ids = np.array([0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 11, 12, 11, 12, 13, 14], dtype=np.int)

    def _get_ann_file_keypoint(self):
        """ self.data_path / annotations / person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in self.image_set_name else 'image_info'
        return os.path.join(self.dataset_path, 'annotations',
                            prefix + '_' + self.image_set_name + '.json')

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.data_name:
            file_name = 'COCO_%s_' % self.data_name + file_name
        image_path = os.path.join(self.dataset_path, 'images', self.data_name, file_name)

        return image_path

    def _xywh2cs(self, x, y, w, h):
        c_x = x + w * 0.5
        c_y = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        if c_x != -1:
            w = w * 1.25
            h = h * 1.25

        return c_x, c_y, w, h

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def jnt_bbox_db(self):
        return self.gt_db()

    def gt_db(self):
        _gt_db = self._load_coco_keypoint_annotations()
        self.all_boxes = []
        for i in range(0, len(_gt_db)):
            self.all_boxes.append({
                'category_id': _gt_db[i]['category_id'],
                'image_id': _gt_db[i]['image_id'],
                'bbox': _gt_db[i]['bbox'],
                'score': _gt_db[i]['score']
            })
        return _gt_db

    def dt_db(self, dt_file):
        return self._load_coco_person_detection_results(dt_file)

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        file_name = '{}_gt_keypoint_db.pkl'.format(self.name)
        cache_file = os.path.join(self.cache_path, file_name)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pickle.load(fid)
            print('{} gt db loaded from {}'.format(self.name, cache_file))
            return db

        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_db, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt db to {}'.format(cache_file))

        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.joint_num, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.joint_num, 3), dtype=np.float)
            for ipt in range(self.joint_num):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            c_x, c_y, w, h = self._box2cs(obj['clean_bbox'][:4])

            if np.sum(joints_3d_vis[:, 0]) < 2:
                continue

            rec.append({
                'image': self.image_path_from_index(index),
                'center_x': c_x,
                'center_y': c_y,
                'width': w,
                'height': h,
                'flip_pairs': self.flip_pairs,
                'parent_ids': self.parent_ids,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'category_id': obj['category_id'],
                'image_id': obj['image_id'],
                'bbox': obj['clean_bbox'],
                'score': 1.0
            })

        return rec

    def _load_coco_person_detection_results(self, dt_file):
        with open(dt_file, 'r') as f:
            self.all_boxes = json.load(f)

        if not self.all_boxes:
            print('Load json fail!')
            return None

        print('Total boxes: {}'.format(len(self.all_boxes)))

        kpt_db = []
        num_boxes = 0
        filter_all_boxes = []
        for n_img in range(0, len(self.all_boxes)):
            det_res = self.all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1
            filter_all_boxes.append(det_res)

            c_x, c_y, w, h = self._box2cs(box)
            joints_3d = np.zeros((self.joint_num, 3), dtype=np.float)
            joints_3d_vis = np.ones((self.joint_num, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center_x': c_x,
                'center_y': c_y,
                'width': w,
                'height': h,
                'flip_pairs': self.flip_pairs,
                'parent_ids': self.parent_ids,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        self.all_boxes = filter_all_boxes

        print('Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def evaluate(self, preds, result_path):
        preds_2d = preds[:, :, 0:2]
        preds_score = preds[:, :, 3:4]
        preds = np.concatenate((preds_2d, preds_score), axis=2)

        res_folder = os.path.join(result_path, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.image_set_name)

        # person x (keypoints)
        _kpts = []
        # print 'len keypoints', len(keypoints)
        for idx, kpt in enumerate(preds):
            bbx = self.all_boxes[idx]
            x, y, w, h = bbx['bbox'][:4]
            _kpts.append({
                'keypoints': kpt,
                # 'center': all_boxes[idx][0:2],
                # 'scale': all_boxes[idx][2:4],
                'area': w * h,
                'score': bbx['score'],
                'image': int(bbx['image_id'])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.joint_num
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score
            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
                           oks_thre)
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        # print 'len oks nmsed kpts', len(oks_nmsed_kpts)
        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        if 'test' not in self.image_set_name:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            return info_str

    def _write_coco_keypoint_results(self, keypoints, res_file):

        # print self.image_set_index[:10]
        data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                      'cls_ind': cls_ind,
                      'cls': cls,
                      'ann_type': 'keypoints',
                      'keypoints': keypoints
                      }
                     for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        print('Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            # print len(_key_points), _key_points.shape
            key_points = np.zeros(
                (_key_points.shape[0], self.joint_num * 3), dtype=np.float)

            for ipt in range(self.joint_num):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [{'image_id': img_kpts[k]['image'],
                       'category_id': cat_id,
                       'keypoints': list(key_points[k]),
                       'score': img_kpts[k]['score'],
                       # 'center': list(img_kpts[k]['center']),
                       # 'scale': list(img_kpts[k]['scale'])
                       } for k in range(len(img_kpts))]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        info_str = coco_eval.summarize()

        eval_file = os.path.join(
            res_folder, 'keypoints_%s_results.pkl' % self.image_set_name)

        with open(eval_file, 'wb') as f:
            pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
        print('coco eval results saved to %s' % eval_file)

        name_value = [
            ('AP', coco_eval.stats[0]),
            ('AP50', coco_eval.stats[1]),
            ('AP75', coco_eval.stats[2]),
            ('APm', coco_eval.stats[3]),
            ('APl', coco_eval.stats[4]),
            ('AR', coco_eval.stats[5]),
            ('AR50', coco_eval.stats[6]),
            ('AR75', coco_eval.stats[7]),
            ('ARm', coco_eval.stats[8]),
            ('ARl', coco_eval.stats[9])
        ]
        return name_value