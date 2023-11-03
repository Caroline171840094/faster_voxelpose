# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json_tricks as json
import pickle
import logging
import cv2
import copy
from utils.hi4d_cam_utils import read_cameras

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    'pair00_dance00',
    'pair00_fight00',
    'pair00_highfive00',
    'pair01_basketball01',
    'pair01_fight01',
    'pair01_highfive01',
    'pair02_backhug02',
    'pair02_dance02',
    'pair02_hug02'
]
VAL_LIST = [
    'pair00_taichi00',
    'pair00_yoga00',
    'pair01_hug01',
    'pair01_talk01',
    'pair02_pose02',
]

# hi4d_joints_def = {
#     'pelvis': 0,
#     'neck': 1,
#     'right-hip': 2,
#     'left-hip': 3,
#     'nose': 4,
#     'right_shoulder': 5,
#     'left_shoulder': 6,
#     'right_knee': 7,
#     'left_knee': 8,
#     'rear': 9,
#     'lear': 10,
#     'right_elbow': 11,
#     'left_elbow': 12,
#     'right_ankle': 13,
#     'left_ankle': 14,
#     'right_wrist': 15,
#     'left_wrist': 16,
#     'LBigToe': 17,
#     'RBigToe': 18,
# }

# hi4d_bones_def = [
#     [0, 1], [0, 2],  # trunk
#     [0, 3], [3, 4], [4, 5],  # left arm
#     [0, 9], [9, 10], [10, 11],  # right arm
#     [2, 6], [6, 7], [7, 8],  # left leg
#     [2, 12], [12, 13], [13, 14],  # right leg
# ]

hi4d_joints_def = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}
hi4d_bones_def = [
    [13, 12],  # head
    [12, 9], [9, 10], [10, 11],  # left arm
    [12, 8], [8, 7], [7, 6],  # right arm
    [9, 3], [8, 2],  # trunk
    [3, 4], [4, 5],  # left leg
    [2, 1], [1, 0],  # right leg
]
hi4d_to_panoptic = [1,4,0,6,12,16,3,8,14, 5,11,15,2,7,13]

class Hi4D(JointsDataset):
    def __init__(self, cfg, is_train=True, transform=None):
        super().__init__(cfg, is_train, transform)
        
        self.num_joints = len(hi4d_joints_def)
        self.num_views = cfg.DATASET.CAMERA_NUM
        self.root_id = cfg.DATASET.ROOT_JOINT_ID

        self.has_evaluate_function = True
        self.transform = transform
        self.cam_list = ['4', '16', '28', '40', '52', '64', '76', '88'][:self.num_views]

        if is_train:
            self.image_set = 'train'
            self.sequence_list = TRAIN_LIST
            self._interval = 1
        else:
            self.image_set = 'validation'
            self.sequence_list = VAL_LIST
            self._interval = 4

        self.cameras = self._get_cam()
        self.db_file = '{}_meta.pkl'.format(self.image_set, self.num_views)
        self.db_file = osp.join(self.dataset_dir, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, 'rb'))
            assert info['sequence_list'] == self.sequence_list
            assert info['interval'] == self._interval
            self.db = info['db']
        else:
            self._get_db()
            info = {
                'sequence_list': self.sequence_list,
                'interval': self._interval,
                'db': self.db
            }
            pickle.dump(info, open(self.db_file, 'wb'))
        self.db_size = len(self.db)
    
    def _get_db(self):
        for seq in self.sequence_list:
            curr_anno = osp.join(self.dataset_dir, seq, 'skel19_gt')
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))
            for i, anno_file in enumerate(anno_files):
                if i % self._interval == 0:
                    with open(anno_file, "r") as f:
                        data_kpts = json.load(f)
                    if len(data_kpts) == 0:
                        continue
                    
                    all_image_path = []
                    missing_image = False
                    framename = osp.basename(anno_file).replace("json", 'jpg')
                    for k in range(self.num_views):
                        image_path = osp.join(self.dataset_dir, seq, "images_resized", self.cam_list[k], framename)
                        if not osp.exists(image_path):
                            logger.info("Image not found: {}. Skipped.".format(image_path))
                            missing_image = True
                            break
                        all_image_path.append(image_path)
                        
                    
                    if missing_image:
                        continue
                
                    all_poses_3d = []
                    all_poses_3d_vis = []
                    for kpts in data_kpts:
                        pose3d = np.array(kpts['keypoints3d']).reshape((-1, 3))
                        pose3d = pose3d[hi4d_to_panoptic, :]
                        pose3d = pose3d[:self.num_joints]

                        joints_vis = np.ones_like(pose3d[:, -1])
                        # ignore the joints with visibility less than 0.1
                        if joints_vis[self.root_id] <= 0.1:
                            continue

                        all_poses_3d.append(pose3d[:, 0:3] * 1000.0)
                        all_poses_3d_vis.append(joints_vis)
                        
                    if len(all_poses_3d) > 0:
                        self.db.append({
                            'seq': seq,
                            'all_image_path': all_image_path,
                            'joints_3d': all_poses_3d,
                            'joints_3d_vis': all_poses_3d_vis,
                        })
        super()._rebuild_db()
        logger.info("=> {} images from {} views loaded".format(len(self.db), self.num_views))
        return

    def _get_cam(self):
        cameras = dict()
        for seq in self.sequence_list:
            cameras[seq] = []
            cams_dict = read_cameras(osp.join(self.dataset_dir, seq))
            cams_new = []
            for cam in self.cam_list:
                cam_info = {}
                cam_info['R'] = cams_dict[cam]['R']
                cam_info['T'] = cams_dict[cam]['T']
                cam_info['fx'] = cams_dict[cam]['K'][0, 0]
                cam_info['fy'] = cams_dict[cam]['K'][1, 1]
                cam_info['cx'] = cams_dict[cam]['K'][0, 2]
                cam_info['cy'] = cams_dict[cam]['K'][1, 2]
                cam_info['k'] = cams_dict[cam]['dist'][0, [0, 1, 4]].reshape(3, 1)
                cam_info['p'] = cams_dict[cam]['dist'][0, [2, 3]].reshape(2, 1)
                cams_new.append(cam_info)
            cameras[seq] = cams_new
        return cameras

    def __getitem__(self, idx):
        input, target, meta, input_heatmap = super().__getitem__(idx)
        return input, target, meta, input_heatmap

    def __len__(self):
        return self.db_size

    def evaluate(self, preds):
        eval_list = []
        gt_num = self.db_size
        import ipdb;ipdb.set_trace()
        assert len(preds) == gt_num, 'number mismatch'
        
        total_gt = 0
        for i in range(len(preds)):
            db_rec = copy.deepcopy(self.db[i])
            num_person = db_rec['meta']['num_person']
            joints_3d = db_rec['meta']['joints_3d'][:num_person]
            joints_3d_vis = db_rec['meta']['joints_3d_vis'][:num_person]

            if num_person == 0:
                continue

            pred = preds[i].detach().cpu().numpy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis > 0.1
                    mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })
            total_gt += len(joints_3d)

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)

        mpjpe = self._eval_list_to_mpjpe(eval_list)
        recall = self._eval_list_to_recall(eval_list, total_gt)
        msg = 'Evaluation results on Panoptic dataset:\nap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t' \
              'ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t' \
              'recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}'.format(
                aps_25=aps[0], aps_50=aps[1], aps_75=aps[2], aps_100=aps[3],
                aps_125=aps[4], aps_150=aps[5], recall=recall, mpjpe=mpjpe
              )
        metric = np.mean(aps)

        return metric, msg

    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

        return len(np.unique(gt_ids)) / total_gt