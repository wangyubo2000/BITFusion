# modify from https://github.com/mit-han-lab/bevfusion
from typing import Any, Dict
import os
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image
from torch import Tensor
from mmdet3d.datasets import GlobalRotScaleTrans
from mmdet3d.registry import TRANSFORMS
from typing import Dict, List, Optional, Tuple, Union
import math
# tta in model
from mmengine.model import BaseTTAModel
from mmdet3d.registry import MODELS
from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d
from mmdet3d.structures import (BaseInstance3DBoxes, Det3DDataSample,
                                xywhr2xyxyr)
from mmengine.structures import InstanceData

@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_matrix'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVFusionRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = np.eye(4)
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        return data


@TRANSFORMS.register_module()
class BEVFusionGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS)."""

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._trans_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']

        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = np.eye(4)
        input_dict[
            'lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']

        return input_dict


@TRANSFORMS.register_module()
class GridMask(BaseTransform):

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def transform(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results
@MODELS.register_module()
class Det3DTTAModel(BaseTTAModel):
    """Test-time augmentation model for LiDAR / BEVFusion 3D detection."""

    def __init__(self, tta_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.tta_cfg = tta_cfg
    def test_step(self, data):

        aug_outputs = []
        for idx, d in enumerate(data):
            # Perform the augmentation
            augmented_data = self.module.test_step(d)
            # Debugging: Print the type of pred_instances_3d
            # Assuming augmented_data is a list of Det3DDataSample objects
            # for idx, sample in enumerate(augmented_data):
            #     # 确保 sample 是 Det3DDataSample 类型
            #     if isinstance(sample, Det3DDataSample):
            #         print(f"Pred 3D BBox for sample {idx + 1}: {sample.pred_instances_3d.bboxes_3d}")
            #         print(f"Pred 3D scores: {sample.pred_instances_3d.scores_3d}")
            #     else:
            #         print(f"Sample {idx + 1} is not a Det3DDataSample object.")
            aug_outputs.append(augmented_data)

        # Ensure that the outputs are wrapped correctly when batch_size=1
        if isinstance(aug_outputs[0], Det3DDataSample):
            aug_outputs = [[x] for x in aug_outputs]

        # Flatten the augmented outputs into a 2D list for merging
        data_list = [[_data[idx] for _data in aug_outputs] for idx in range(len(aug_outputs[0]))]

        # Merge the predictions
        merged = self.merge_preds(data_list)
        return merged

    def merge_preds(self, data_samples_list: List[List[Det3DDataSample]]):
        """合并增强后的预测结果"""
        merged_samples = []
        for data_samples in data_samples_list:
            aug_results, aug_metas = [], []
            for idx,sample in enumerate(data_samples):
                preds = sample.pred_instances_3d
                aug_results.append(dict(
                    bbox_3d=preds.bboxes_3d,
                    scores_3d=preds.scores_3d,
                    labels_3d=preds.labels_3d))
                aug_metas.append(sample.metainfo)
            merged_result = merge_aug_bboxes_3d(
                aug_results=aug_results,
                aug_batch_input_metas=aug_metas,
                test_cfg=self.tta_cfg)
            inst = InstanceData()
            inst.bboxes_3d = merged_result['bboxes_3d']
            inst.scores_3d = merged_result['scores_3d']
            inst.labels_3d = merged_result['labels_3d']

            merged_sample = Det3DDataSample()
            merged_sample.set_metainfo(data_samples[0].metainfo)   # idx_info
            merged_sample.pred_instances_3d = inst
            merged_sample.pred_instances = InstanceData()
            merged_samples.append(merged_sample)
        return merged_samples

@TRANSFORMS.register_module()
class NormalizeCam2ImgToK(BaseTransform):
    """把样本里每个相机的 cam2img 规整为 3x3 K。"""
    def transform(self, results):
        images = results.get('images', {})
        for _, cam_item in images.items():
            K = np.array(cam_item.get('cam2img'), dtype=np.float32)
            if K.shape == (3, 4):      # 投影矩阵 P
                K = K[:, :3]
            elif K.shape == (4, 4):    # 齐次写法
                K = K[:3, :3]
            elif K.shape != (3, 3):
                raise ValueError(f"Unexpected cam2img shape: {K.shape}")
            cam_item['cam2img'] = K.tolist()  # 回写为 3x3
        return results

@TRANSFORMS.register_module()
class AttachCamFolder(BaseTransform):
    """给 images[*]['img_path'] 补相机子目录前缀，不改文件名。"""
    def __init__(self, cam_prefixes: dict, base_dir: str = ''):
        self.cam_prefixes = cam_prefixes  # 例如 {'CAM_FRONT': 'image_1', ...}
        self.base_dir = base_dir          # 留空则依赖 data_root

    def transform(self, results):
        images = results.get('images', {})
        for cam, cam_item in images.items():
            fname = cam_item.get('img_path', '')
            # 只处理“纯文件名”，已有子目录或绝对路径一律不动
            if fname and not os.path.isabs(fname) and os.path.dirname(fname) == '':
                sub = self.cam_prefixes.get(cam, '')
                rel = os.path.join(sub, fname) if sub else fname
                cam_item['img_path'] = os.path.join(self.base_dir, rel) if self.base_dir else rel
        return results
