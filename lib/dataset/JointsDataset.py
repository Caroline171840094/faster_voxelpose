# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

from utils.transforms import get_affine_transform, affine_transform, get_scale
from utils.cameras_cpu import project_pose

logger = logging.getLogger(__name__)

class JointsDataset(Dataset):
    def __init__(self, cfg, is_train=True, add_noise_to_heatmap=False, transform=None):
        self.cfg = cfg
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE
        self.num_views = cfg.DATASET.CAMERA_NUM
        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.distort = cfg.DISTORT_IMAGE
        self.dataset_root = self._get_dataset_root_dir(cfg)

        # relates to camera calibration
        self.ori_image_width = cfg.DATASET.ORI_IMAGE_WIDTH
        self.ori_image_height = cfg.DATASET.ORI_IMAGE_HEIGHT

        self.target_type = cfg.NETWORK.TARGET_TYPE
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.sigma = cfg.NETWORK.SIGMA
    
        self.space_size = np.array(cfg.CAPTURE_SPEC.SPACE_SIZE)
        self.space_center = np.array(cfg.CAPTURE_SPEC.SPACE_CENTER)
        self.voxels_per_axis = np.array(cfg.CAPTURE_SPEC.VOXELS_PER_AXIS)
        self.individual_space_size = np.array(cfg.INDIVIDUAL_SPEC.SPACE_SIZE)

        self.random_scale_heatmap = cfg.SYNTHETIC.RANDOM_SCALE_HEATMAP
        self.random_erase_heatmap = cfg.SYNTHETIC.RANDOM_ERASE_HEATMAP

        if is_train:
            self.input_heatmap_src = cfg.DATASET.TRAIN_HEATMAP_SRC
        else:
            self.input_heatmap_src = cfg.DATASET.TEST_HEATMAP_SRC
        self.transform = transform
        self.add_noise_to_heatmap = add_noise_to_heatmap
        self.cameras = None
        self.db = []

    def _get_dataset_root_dir(self, cfg):
        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, '../..', cfg.DATASET.ROOT)
        dataset_root = os.path.abspath(dataset_root)
        return dataset_root

    def _rebuild_db(self):
        for idx in range(len(self.db)):
            db_rec = self.db[idx]

            r = 0
            c = np.array([self.ori_image_width / 2.0, self.ori_image_height / 2.0])
            s = get_scale((self.ori_image_width, self.ori_image_height),
                        self.image_size)
            trans = get_affine_transform(c, s, r, self.image_size)
            
            if self.input_heatmap_src == 'image':
                input_heatmap = torch.zeros(1, 1, 1)
            
            elif self.input_heatmap_src == 'pred':
                assert 'pred_pose2d' in db_rec and db_rec['pred_pose2d'] is not None, 'dataset must provide pred_pose2d'
                pred_pose2d = db_rec['pred_pose2d']
                for n in range(len(pred_pose2d)):
                    for i in range(len(pred_pose2d[n])):
                        pred_pose2d[n][i, 0:2] = affine_transform(
                            pred_pose2d[n][i, 0:2], trans)
                input_heatmap = self.generate_input_heatmap(pred_pose2d)
                input_heatmap = torch.from_numpy(input_heatmap)

            elif self.input_heatmap_src == 'gt':
                assert 'joints_3d' in db_rec, 'dataset must provide gt joints_3d'
                joints_3d = db_rec['joints_3d']
                joints_3d_vis = db_rec['joints_3d_vis']
                nposes = len(joints_3d)
                cam = db_rec['camera']
                joints_2d = []
                joints_2d_vis = []
                for n in range(nposes):
                    x = project_pose(joints_3d[n], cam, distort=self.distort)

                    x_check = np.bitwise_and(x[:, 0] >= 0,
                                            x[:, 0] <= self.ori_image_width - 1)
                    y_check = np.bitwise_and(x[:, 1] >= 0,
                                            x[:, 1] <= self.ori_image_height - 1)
                    check = np.bitwise_and(x_check, y_check)
                    vis = joints_3d_vis[n] > 0
                    vis[np.logical_not(check)] = 0
                    joints_2d.append(x)
                    joints_2d_vis.append(
                        np.repeat(np.reshape(vis, (-1, 1)), 2, axis=1))

                for n in range(nposes):
                    for i in range(len(joints_2d[n])):
                        if joints_2d_vis[n][i, 0] > 0.0:
                            joints_2d[n][i, 0:2] = affine_transform(
                                joints_2d[n][i, 0:2], trans)
                            if (np.min(joints_2d[n][i, :2]) < 0
                                    or joints_2d[n][i, 0] >= self.image_size[0]
                                    or joints_2d[n][i, 1] >= self.image_size[1]):
                                joints_2d_vis[n][i, :] = 0
                input_heatmap = self.generate_input_heatmap(joints_2d)
                input_heatmap = torch.from_numpy(input_heatmap)

            # dataset only for testing: no gt 3d pose
            if 'joints_3d' not in db_rec:
                meta = {
                    'key': db_rec['key'],
                    'image': db_rec['image'],
                    'camera': db_rec['camera']
                }
                target = np.zeros((1, 1, 1), dtype=np.float32)
                target = torch.from_numpy(target)
                self.db[idx] = {
                    'target': target,
                    'input_heatmap': input_heatmap,
                    'meta': meta
                }
                continue

            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_vis']
            nposes = len(joints_3d)
            assert nposes <= self.max_people, 'too many persons'

            joints_3d_u = np.zeros((self.max_people, self.num_joints, 3))
            joints_3d_vis_u = np.zeros((self.max_people, self.num_joints))
            for i in range(nposes):
                joints_3d_u[i] = joints_3d[i][:, 0:3]
                joints_3d_vis_u[i] = joints_3d_vis[i]

            if isinstance(self.root_id, int):
                roots_3d = joints_3d_u[:, self.root_id]
            elif isinstance(self.root_id, list):
                roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
            
            target = self.generate_target(joints_3d, joints_3d_vis)
            meta = {
                'num_person': nposes,
                'joints_3d': joints_3d_u,
                'joints_3d_vis': joints_3d_vis_u,
                'roots_3d': roots_3d,
                'bbox': target['bbox'],
                'key': db_rec['key'],
                'camera': db_rec['camera'],
                'trans': trans
            }
            if 'image' in db_rec.keys():
                meta['image'] = db_rec['image']
                
            self.db[idx] = {
                'target': target,
                'input_heatmap': input_heatmap,
                'meta': meta
            }
        return

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = self.db[idx]

        if self.input_heatmap_src == 'image':
            image_file = db_rec['meta']['image']
            input = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            assert input is not None, "image file not exist"
            if self.color_rgb:
                input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            
            '''
            input = cv2.warpAffine(
                data_numpy,
                trans, (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)
            '''
            
            if self.transform:
                input = self.transform(input)
        else:
            input = np.zeros((1, 1, 1), dtype=np.float32)

        target = db_rec["target"]
        meta = db_rec["meta"]
        input_heatmap = db_rec["input_heatmap"]
        return input, target, meta, input_heatmap

    def compute_human_scale(self, pose, joints_vis):
        idx = (joints_vis > 0.1)
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        return np.clip(np.maximum(maxy - miny, maxx - minx)**2,  1.0 / 4 * 96**2, 4 * 96**2)

    def generate_target(self, joints_3d, joints_3d_vis):
        # return: index, offset, bbox, 2d_heatmaps, 1d_heatmaps
        num_people = len(joints_3d)
        space_size = np.array(self.space_size)
        space_center = np.array(self.space_center)
        individual_space_size = np.array(self.individual_space_size)
        voxels_per_axis = np.array(self.voxels_per_axis)
        voxel_size = space_size / (voxels_per_axis - 1)

        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, voxels_per_axis[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, voxels_per_axis[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, voxels_per_axis[2]) + space_center[2]

        target_index = np.zeros((self.max_people))
        target_2d = np.zeros((voxels_per_axis[0], voxels_per_axis[1]), dtype=np.float32)
        target_1d = np.zeros((self.max_people, voxels_per_axis[2]), dtype=np.float32)
        target_bbox = np.zeros((self.max_people, 2), dtype=np.float32)
        target_offset = np.zeros((self.max_people, 2), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # mid-hip
            idx = (joints_3d_vis[n] > 0.1)
            if isinstance(joint_id, int):
                center_pos = joints_3d[n][joint_id]
            elif isinstance(joint_id, list):
                center_pos = (joints_3d[n][joint_id[0]] + joints_3d[n][joint_id[1]]) / 2.0
            
            # compute target index, offset and bbox size
            loc = (center_pos - space_center + 0.5 * space_size) / voxel_size
            assert np.sum(loc < 0) == 0 and np.sum(loc > voxels_per_axis) == 0, "human centers out of bound!" 
            # flatten 2d index
            target_index[n] = (loc // 1)[0] * voxels_per_axis[0] + (loc // 1)[1]
            target_offset[n] = (loc % 1)[:2]
            target_bbox[n] = ((2 * np.abs(center_pos - joints_3d[n][idx]).max(axis = 0) + 200.0) / individual_space_size)[:2]
            if np.sum(target_bbox[n] > 1) > 0:
                print("the bounding box isn't sufficiently large to cover poses!")
            # assert np.sum(target_bbox[n] > 1) == 0 , "the bounding box isn't sufficiently large to cover poses!"

            # Gaussian distribution
            mu_x, mu_y, mu_z = center_pos[0], center_pos[1], center_pos[2]
            i_x = [np.searchsorted(grid1Dx,  mu_x - 3 * cur_sigma),
                       np.searchsorted(grid1Dx,  mu_x + 3 * cur_sigma, 'right')]
            i_y = [np.searchsorted(grid1Dy,  mu_y - 3 * cur_sigma),
                       np.searchsorted(grid1Dy,  mu_y + 3 * cur_sigma, 'right')]
            i_z = [np.searchsorted(grid1Dz,  mu_z - 3 * cur_sigma),
                       np.searchsorted(grid1Dz,  mu_z + 3 * cur_sigma, 'right')]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            # generate 2d target
            gridx, gridy = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], indexing='ij')
            g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2) / (2 * cur_sigma ** 2))
            target_2d[i_x[0]:i_x[1], i_y[0]:i_y[1]] = np.maximum(target_2d[i_x[0]:i_x[1], i_y[0]:i_y[1]], g)

            # generate 1d target
            gridz = grid1Dz[i_z[0]:i_z[1]]
            g = np.exp(-(gridz - mu_z) ** 2 / (2 * cur_sigma ** 2))
            target_1d[n, i_z[0]:i_z[1]] = np.maximum(target_1d[n, i_z[0]:i_z[1]], g)
            
        target_2d = np.clip(target_2d, 0, 1)
        target_1d = np.clip(target_1d, 0, 1)
        mask = (np.arange(self.max_people) <= num_people)
        target = {'index': target_index, 'offset': target_offset, 'bbox': target_bbox,
                  '2d_heatmaps': target_2d, '1d_heatmaps': target_1d, 'mask':mask}
        return target

    def generate_input_heatmap(self, joints):
        nposes = len(joints)
        num_joints = self.cfg.DATASET.NUM_JOINTS

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(
                        joints[n][:, 0:2] / feat_stride, np.ones(num_joints))
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2

                    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                    # add noise to heatmap
                    if self.add_noise_to_heatmap:
                        # random scaling
                        if self.random_scale_heatmap:
                            scale = np.random.uniform(0.4, 0.9)
                            g *= scale
                            '''
                            scale = 0.9 + np.random.randn(1) * 0.03 if random.random() < 0.6 else 1.0
                            if joint_id in [7, 8]:
                                scale = scale * 0.5 if random.random() < 0.1 else scale
                            elif joint_id in [9, 10]:
                                scale = scale * 0.2 if random.random() < 0.1 else scale
                            else:
                                scale = scale * 0.5 if random.random() < 0.05 else scale
                            g *= scale
                            '''
                        # random occlusion
                        if self.random_erase_heatmap:
                            start = [int(np.random.uniform(0, self.heatmap_size[1] -1)),
                                     int(np.random.uniform(0, self.heatmap_size[0] -1))]
                            end = [int(min(start[0] + np.random.uniform(self.heatmap_size[1] / 4, 
                                   self.heatmap_size[1] * 0.75), self.heatmap_size[1])),
                                   int(min(start[1] + np.random.uniform(self.heatmap_size[0] / 4,
                                   self.heatmap_size[0] * 0.75), self.heatmap_size[0]))]
                            g[start[0]:end[0], start[1]:end[1]] = 0.0

                    g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1],
                                     img_x[0]:img_x[1]] = np.maximum(
                                         target[joint_id][img_y[0]:img_y[1],
                                                          img_x[0]:img_x[1]],
                                         g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        return target