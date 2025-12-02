from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        cam_guided_plugin: Optional[dict] = None, #camera->lidar指导模块
        lidar_guided_plugin: Optional[dict] = None, #lidar->camera 指导模块
        mssa_plugin: Optional[dict] = None,         #单帧空间加强
        memory_bank: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)    # 体素化

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder) #体素编码->bev了

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.cam_guided = MODELS.build(
            cam_guided_plugin) if cam_guided_plugin is not None else None
        self.lidar_guided = MODELS.build(
            lidar_guided_plugin) if lidar_guided_plugin is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.mssa_plugin = MODELS.build(mssa_plugin) if mssa_plugin is not None else None
        self.memory_bank = MODELS.build(memory_bank) if memory_bank is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)
        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def _tokens_in_rotated_boxes_ratio(self, indices_list, boxes_list,
                                       x_range, y_range, Hb, Wb, device,
                                       try_swap=True, use_aabb_fallback=True):
        import torch
        dx = (x_range[1] - x_range[0]) / float(Wb)
        dy = (y_range[1] - y_range[0]) / float(Hb)

        hit, tot = 0.0, 0.0
        for b, ind in enumerate(indices_list):
            if ind is None or ind.numel() == 0:
                continue
            i = ind[:, 0].long().clamp(0, Hb - 1)
            j = ind[:, 1].long().clamp(0, Wb - 1)
            # 栅格中心物理坐标
            x_c = x_range[0] + (j.float() + 0.5) * dx
            y_c = y_range[0] + (i.float() + 0.5) * dy
            N = x_c.numel()
            tot += float(N)

            boxes = boxes_list[b]
            boxes = (boxes.tensor if hasattr(boxes, 'tensor') else boxes)
            if boxes is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(x_c.device).float()

            # 计算一次命中（旋转矩形），若需要可交换 dx/dy（很多数据集 dx=length(沿x), dy=width(沿y)，
            # 但也有人反过来；再加上 yaw 朝向可能差 π/2，稳妥起见试两套）
            def rotated_hit(xc, yc, boxes, swap_lw=False):
                x0, y0 = boxes[:, 0], boxes[:, 1]
                if swap_lw:
                    l, w = boxes[:, 4], boxes[:, 3]  # 交换
                else:
                    l, w = boxes[:, 3], boxes[:, 4]
                yaw = boxes[:, 6]
                cos, sin = torch.cos(yaw), torch.sin(yaw)  # [M]
                M = boxes.shape[0]

                xc = xc.view(1, N) - x0.view(M, 1)  # [M,N]
                yc = yc.view(1, N) - y0.view(M, 1)  # [M,N]
                xp = cos.view(M, 1) * xc + sin.view(M, 1) * yc
                yp = -sin.view(M, 1) * xc + cos.view(M, 1) * yc

                in_x = xp.abs() <= (l.view(M, 1) * 0.5 + 1e-6)
                in_y = yp.abs() <= (w.view(M, 1) * 0.5 + 1e-6)
                inside_any = (in_x & in_y).any(dim=0)  # [N]
                return inside_any

            inside_any = rotated_hit(x_c, y_c, boxes, swap_lw=False)
            if try_swap:
                inside_any_swap = rotated_hit(x_c, y_c, boxes, swap_lw=True)
                inside_any = inside_any | inside_any_swap

            # 兜底：若依然全 0，用 AABB（无旋转），可以暴露“旋转/轴向不一致”的问题
            if use_aabb_fallback and not bool(inside_any.any()):
                x0 = (boxes[:, 0] - boxes[:, 3] * 0.5).view(-1, 1)
                x1 = (boxes[:, 0] + boxes[:, 3] * 0.5).view(-1, 1)
                y0 = (boxes[:, 1] - boxes[:, 4] * 0.5).view(-1, 1)
                y1 = (boxes[:, 1] + boxes[:, 4] * 0.5).view(-1, 1)
                in_x = (x_c.view(1, N) >= x0) & (x_c.view(1, N) <= x1)
                in_y = (y_c.view(1, N) >= y0) & (y_c.view(1, N) <= y1)
                inside_any = (in_x & in_y).any(dim=0)

            hit += float(inside_any.sum().item())

        return hit / max(tot, 1.0)

    def _rasterize_bev_boxes(self, bboxes_3d, x_range, y_range, Hb, Wb, device):
        """轴对齐近似，把 3D 框投到 BEV 栅格，返回 [1,Hb,Wb] bool 掩码。"""
        if hasattr(bboxes_3d, 'tensor'):
            boxes = bboxes_3d.tensor.to(device)
        else:
            boxes = bboxes_3d.to(device)
        if boxes.numel() == 0:
            return torch.zeros(1, Hb, Wb, dtype=torch.bool, device=device)

        x, y, l, w = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4]
        dx = (x_range[1] - x_range[0]) / float(Wb)
        dy = (y_range[1] - y_range[0]) / float(Hb)

        x0 = ((x - l / 2 - x_range[0]) / dx).long().clamp(0, Wb - 1)
        x1 = ((x + l / 2 - x_range[0]) / dx).long().clamp(0, Wb - 1)
        y0 = ((y - w / 2 - y_range[0]) / dy).long().clamp(0, Hb - 1)
        y1 = ((y + w / 2 - y_range[0]) / dy).long().clamp(0, Hb - 1)

        mask = torch.zeros(1, Hb, Wb, dtype=torch.bool, device=device)
        for i in range(boxes.shape[0]):
            mask[:, y0[i]:y1[i] + 1, x0[i]:x1[i] + 1] = True
        return mask

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None


    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        feats = self.img_neck(x)
        if isinstance(feats, (list, tuple)):
            p3_bn = feats[0]    # 小目标检测的特征 [6, 256, 32, 88]
        else:
            p3_bn = feats  # 你的 FPN 特殊实现也可能直接是单层

        # 还需要把给 view_transform 的特征准备好（原逻辑保持）
        # 旧代码里是：如果不是 Tensor 就 feats = feats[0]，这里我们单独取一份供 LSS
        feats_for_lss = feats[0] if not isinstance(feats, torch.Tensor) else feats
        BN, Cn, Hn, Wn = feats_for_lss.size()
        feats_for_lss = feats_for_lss.view(B, int(BN / B), Cn, Hn, Wn) # 用来转换的特征  [1, 6, 256, 32, 88]
        # print('shape of feats_for_lss', feats_for_lss.shape)
        # ★★★ 关键：缓存 P3 / K / T（按 [B,N,...] 组织），供 ConvFuser 之前使用
        # p3_bn: [B*N, C, Hf, Wf]  ->  [B, N, C, Hf, Wf]
        Cp3, Hp3, Wp3 = p3_bn.shape[1], p3_bn.shape[2], p3_bn.shape[3]
        p3_bnc = p3_bn.view(B, N, Cp3, Hp3, Wp3).contiguous() # [1, 6, 256, 32, 88]
        self._last_cam_p3 = p3_bnc  # 仍在 GPU 上，后面直接用

        # 相机内参/外参（已经是 [B,N,...] 的张量）
        # camera_intrinsics: [B,N,3,3]，camera2lidar: [B,N,4,4]
        self._last_K = camera_intrinsics  # 直接缓存相机内参数矩阵
        self._last_T_cam2lidar = camera2lidar #直接保存
        self._last_num_cams = N #相机的数量

        # 走原来的 DepthLSSTransform 主路
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            img_bev = self.view_transform(
                feats_for_lss,  # [B,N,C,H,W]
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            ) #img_bev
        return img_bev

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res
    def fold_post_into_K(self,K_img, post_rot, post_trans):
        K_aug = post_rot @ K_img
        K_aug[0, 2] += post_trans[0]
        K_aug[1, 2] += post_trans[1]
        return K_aug

    def scale_K_to_feat(self,K_img, H_img, W_img, H_feat, W_feat):
        sx = W_feat / float(W_img)
        sy = H_feat / float(H_img)
        K = K_img.clone()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return K
    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            B_in, N_in, C_in, H_in, W_in = imgs.shape #[256,704] B=1,C=3 RGB
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(
                imgs, deepcopy(points),
                lidar2image, camera_intrinsics, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, batch_input_metas) #已经是img_BEV特征
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)      #这部分是lidar_BEV特征
        if (self.cam_guided is not None) and (self._last_cam_p3 is not None):
            lidar_bev = pts_feature
            B = lidar_bev.shape[0]
            N = self._last_num_cams
            dev, dt = lidar_bev.device, lidar_bev.dtype
            # 逐相机增强（更省显存；也便于不同外参）
            # 已有: lidar_bev, B, N, self._last_cam_p3, self._last_T_cam2lidar, metas
            for n in range(N):
                cam_feat_p3 = self._last_cam_p3[:, n, ...]  # [B,C,Hf,Wf]  [1, 256, 32, 88]
                Hf, Wf = cam_feat_p3.shape[-2:] #【32，88】
                # 从 metas 取原图尺寸/增强矩阵（按你的meta键名）
                Ks_list, Ts_list = [], []
                for b in range(B):
                    # 保证全部在与 lidar_bev 相同的 device/dtype 上
                    K_img = torch.as_tensor(meta['cam2img'][n][:3, :3], device=dev, dtype=torch.float16)
                    post = torch.as_tensor(meta['img_aug_matrix'][n], device=dev, dtype=torch.float16)
                    K_aug = post[:3, :3] @ K_img
                    K_aug[0, 2] += post[0, 3];
                    K_aug[1, 2] += post[1, 3]
                    K_feat = self.scale_K_to_feat(K_aug, H_in, W_in, Hf, Wf)  # 只改 4 个标量
                    Ks_list.append(K_feat)
                    Ts_list.append(self._last_T_cam2lidar[b, n, ...])
                # 直接传“小列表”，cam_guided 内部逐 b 读用，不再保存 self.last_K 等大字段
                lidar_bev = self.cam_guided(lidar_bev, cam_feat_p3, Ks_list, Ts_list, None)
            pts_feature = lidar_bev
        lidar_bev = pts_feature      # 主要是为了BEV的增强
        if self.lidar_guided is not None:
            Hb, Wb = lidar_bev.shape[-2], lidar_bev.shape[-1]
            # 复用 cam_guided 的距离门控（如果有）；没有就传 None
            if hasattr(self, 'cam_guided') and hasattr(self.cam_guided, 'range_gate'):
                range_alpha = self.cam_guided.range_gate(Hb, Wb, dtype=img_feature.dtype, device=img_feature.device)
            else:
                range_alpha = None
            # 用 LiDAR 生成 1 通道门控，乘到 img_bev 上（仅 img 路径反传，显存极省）
            img_feature = self.lidar_guided(img_feature, lidar_bev, range_alpha)
            features[0] = img_feature  # 别忘了把 features 里的 img_feature 更新
        features.append(pts_feature)
        #为了我可视化
        # self.lidar_add_bev = lidar_bev
        # self.camera_add_bev = img_feature
        if self.fusion_layer is not None:
            x = self.fusion_layer(features) #融合之后是BEV特征
        else:
            assert len(features) == 1, features
            x = features[0]
        # print('draw the num x',x)
        # print('draw the shape of x:',x.shape)
        ############ 空间加强+时间融合 ############
        # if hasattr(self, 'mssa_plugin'):
        #     x = self.mssa_plugin(x)
        # if hasattr(self, 'memory_bank'):
        #     x = self.memory_bank(x)  # 时序融合
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)
        aux_cfg = getattr(self, 'aux_loss_cfg', None)
        loss_cov_val = 0.0  # 默认 0，保证日志里总有 loss_cov

        try:
            if aux_cfg is not None and getattr(self, 'cam_guided', None) is not None:
                indices_list = getattr(self.cam_guided, 'last_indices', None)
                bev_hw = getattr(self.cam_guided, 'last_bev_hw', None)

                if indices_list is not None and bev_hw is not None:
                    Hb, Wb = bev_hw
                    x_range = aux_cfg['x_range'];
                    y_range = aux_cfg['y_range']

                    with torch.no_grad():
                        boxes_list = [d.gt_instances_3d.bboxes_3d for d in batch_data_samples]
                        # 你当前用的是 GPU 版本的话就调用 GPU 版；如果你写了 CPU 版，这里也可以换成 CPU 版
                        coverage = self._tokens_in_rotated_boxes_ratio(
                            indices_list, boxes_list, x_range, y_range, Hb, Wb,
                            device=(feats[0].device if isinstance(feats, (list, tuple)) else feats.device)
                        )

                    # 覆盖率可能是 float 或 tensor，这里统一成 float
                    if isinstance(coverage, torch.Tensor):
                        cov_val = float(coverage.detach().mean().cpu())
                    else:
                        cov_val = float(coverage)

                    # 打印一次便于确认（可注释掉）
                    if not hasattr(self, '_aux_debug_once'):
                        print(
                            f'[aux] coverage={cov_val:.4f}  HbWb=({Hb},{Wb})  tokens={[int(v.shape[0]) for v in indices_list][:2]}')
                        self._aux_debug_once = True

                    loss_cov_val = aux_cfg.get('w_cover', 0.05) * (1.0 - cov_val)
        except Exception as e:
            # 出错也不要中断训练；写个提示继续跑
            if not hasattr(self, '_aux_warn_once'):
                print(f'[aux] coverage branch failed: {repr(e)}')
                self._aux_warn_once = True

        base = feats[0] if isinstance(feats, (list, tuple)) else feats
        losses['loss_cov'] = base.new_tensor(loss_cov_val)
        return losses
