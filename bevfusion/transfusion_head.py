# modify from https://github.com/mit-han-lab/bevfusion
import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from torch import nn
from collections import deque
from mmcv.ops import DeformConv2d
from mmdet3d.models import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.models.layers import nms_bev
from mmdet3d.registry import MODELS
from mmdet3d.structures import xywhr2xyxyr
from typing import Optional, Dict, List, Tuple, Union, Any
from mmengine.model import BaseModule
import math
from einops import rearrange
def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y
# ---------- 基础 ----------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x

# ---------- 小目标友好 FEM ----------
class FEMSmallSafe(nn.Module):
    """默认 stride=1，防止暗降分辨率；用非对称核+空洞扩受野。"""
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8,
                 use_dilation=True, dilation_rates=(3,5), downsample_branch=None): # 保持分辨率；scale：残差系数；
        super().__init__()
        self.scale = scale
        inter = max(1, in_planes // map_reduce)
        self.b0 = nn.Sequential(
            BasicConv(in_planes, 2*inter, 1, 1, 0),
            BasicConv(2*inter, 2*inter, 3, 1, 1, relu=False),
        )
        s1 = stride if (downsample_branch == 1) else 1
        self.b1 = nn.Sequential(
            BasicConv(in_planes, inter, 1, 1, 0),
            BasicConv(inter, (inter//2)*3, (1,3), s1, (0,1)),
            BasicConv((inter//2)*3, 2*inter, (3,1), 1, (1,0)),
            BasicConv(2*inter, 2*inter, 3, 1,
                      padding=(dilation_rates[0] if use_dilation else 1),
                      dilation=(dilation_rates[0] if use_dilation else 1),
                      relu=False)
        )
        s2 = stride if (downsample_branch == 2) else 1
        self.b2 = nn.Sequential(
            BasicConv(in_planes, inter, 1, 1, 0),
            BasicConv(inter, (inter//2)*3, (3,1), s2, (1,0)),
            BasicConv((inter//2)*3, 2*inter, (1,3), 1, (0,1)),
            BasicConv(2*inter, 2*inter, 3, 1,
                      padding=(dilation_rates[1] if use_dilation else 1),
                      dilation=(dilation_rates[1] if use_dilation else 1),
                      relu=False)
        )
        self.merge = BasicConv(6*inter, out_planes, 1, 1, 0, relu=False)
        self.short = BasicConv(in_planes, out_planes, 1,
                               stride=(stride if (downsample_branch is not None) else 1),
                               relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.b0(x); x1 = self.b1(x); x2 = self.b2(x)
        out = self.merge(torch.cat([x0, x1, x2], dim=1))
        out = out * self.scale + self.short(x)
        return self.relu(out) # 小目标检测出来的特征 将多尺度特征的最后一层进行特征加强

# ---------- 候选与不确定性 ----------
class SmallObjCandidateHead(nn.Module):
    def __init__(self, in_channels, mid=128):
        super().__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channels, mid, 3, 1, 1),
            BasicConv(mid, mid, 3, 1, 1),
            nn.Conv2d(mid, 1, 1)
        )
    def forward(self, x):  # [B,1,H,W]
        return self.conv(x) #特征的进一步优化 -> 热力图

class UncertaintyHead(nn.Module):
    def __init__(self, in_channels, mid=128):
        super().__init__()
        self.shared = nn.Sequential(
            BasicConv(in_channels, mid, 3, 1, 1),
            BasicConv(mid, mid, 3, 1, 1)
        )
        self.depth_mu = nn.Conv2d(mid, 1, 1)
        self.depth_logvar = nn.Conv2d(mid, 1, 1)
        self.sem_logit = nn.Conv2d(mid, 1, 1)

    def forward(self, x):
        h = self.shared(x)
        d_mu = F.softplus(self.depth_mu(h))   # 深度均值
        d_logv = self.depth_logvar(h)         # 深度置信度
        sem = torch.sigmoid(self.sem_logit(h)) # 语义置信度
        return d_mu, d_logv, sem
class LiteCamProjector(nn.Module):
    def __init__(self, bev_h, bev_w, x_range, y_range, z_range,
                 enable_fp16=True, cache_rays=False):
        super().__init__()
        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
        self.enable_fp16 = enable_fp16
        self.cache_rays = False  # ★ 彻底关；本实现无需射线缓存

    @torch.no_grad()
    def project_tokens(self, pix_uv, depth_mu, K, T_cam2ego, H, W, Hb, Wb,
                       chunk: int = 1024): # K 和 T 我们已经变化成了cam
        """
        极省显存版：按 token 分块，标量方式计算 cam→ego→BEV 索引。
        返回: mask(bool[N]), ij(Long[M,2])；若 M=0, 返回 (全 False, None)
        """
        device = depth_mu.device
        N = pix_uv.shape[0]
        if N == 0:
            return torch.zeros(0, dtype=torch.bool, device=device), None

        # —— 统一 dtype（fp16 优先）——
        compute_dtype = torch.float16 if (self.enable_fp16 and depth_mu.dtype in (torch.float16, torch.bfloat16, torch.float32)) else depth_mu.dtype
        fx = K[0, 0].to(device=device, dtype=compute_dtype)
        fy = K[1, 1].to(device=device, dtype=compute_dtype)
        cx = K[0, 2].to(device=device, dtype=compute_dtype)
        cy = K[1, 2].to(device=device, dtype=compute_dtype)

        R = T_cam2ego[:3, :3].to(device=device, dtype=compute_dtype)
        t = T_cam2ego[:3, 3].to(device=device, dtype=compute_dtype)

        # 预分配输出（避免多次 concat）
        mask_out = torch.zeros(N, dtype=torch.bool, device=device)
        ij_out   = torch.empty(N, 2, dtype=torch.int64, device=device)  # 写指针填充
        write = 0

        # 常量预先转成 compute_dtype，减少隐式 cast 的中间副本
        xr0, xr1 = [torch.tensor(v, device=device, dtype=compute_dtype) for v in self.x_range]
        yr0, yr1 = [torch.tensor(v, device=device, dtype=compute_dtype) for v in self.y_range]
        zr0, zr1 = [torch.tensor(v, device=device, dtype=compute_dtype) for v in self.z_range]
        dx = (xr1 - xr0) / Wb
        dy = (yr1 - yr0) / Hb

        # 分块处理，限制峰值
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            uv  = pix_uv[s:e].to(device=device)
            d   = depth_mu[s:e].to(device=device, dtype=compute_dtype)

            # u,v 限制到特征平面范围（不创建 long 副本，保持 fp 参与计算）
            u = uv[:, 0].clamp_(0, W - 1).to(dtype=compute_dtype)
            v = uv[:, 1].clamp_(0, H - 1).to(dtype=compute_dtype)

            # 相机坐标（Z 深度语义）：X=(u-cx)/fx*d, Y=(v-cy)/fy*d, Z=d
            X = (u - cx) / fx * d
            Y = (v - cy) / fy * d
            Z = d

            # 手写 cam→ego（避免构造 [n,3] 和 matmul 中间量）
            # Xe = R @ [X,Y,Z] + t
            x = R[0,0]*X + R[0,1]*Y + R[0,2]*Z + t[0]
            y = R[1,0]*X + R[1,1]*Y + R[1,2]*Z + t[1]
            z = R[2,0]*X + R[2,1]*Y + R[2,2]*Z + t[2]

            # 立刻做 AABB 过滤（减少后续 idx_add 的尺寸）
            m = (x >= xr0) & (x < xr1) & (y >= yr0) & (y < yr1) & (z >= zr0) & (z < zr1)
            mc = int(m.sum().item())
            if mc == 0:
                continue

            # 映射到 BEV 索引（先 fp 计算，再一次性 long）
            j = ((x[m] - xr0) / dx).floor().clamp_(0, Wb - 1)
            i = ((y[m] - yr0) / dy).floor().clamp_(0, Hb - 1)
            ij = torch.stack([i.to(torch.int64), j.to(torch.int64)], dim=-1)

            # 写入预分配缓冲
            mask_out[s:e][m] = True
            ij_out[write:write + mc] = ij
            write += mc

            # 释放块内临时变量，配合 empty_cache 降碎片
            del uv, d, u, v, X, Y, Z, x, y, z, m, i, j, ij
            torch.cuda.empty_cache()

        if write == 0:
            return mask_out, None
        return mask_out, ij_out[:write]
# ---------- 轻量投影器（稀疏 token，近邻射线） ----------
# class LiteCamProjector(nn.Module):
#     def __init__(self, bev_h, bev_w, x_range, y_range, z_range,
#                  enable_fp16=True, cache_rays=True):
#         super().__init__()
#         self.bev_h, self.bev_w = bev_h, bev_w   # 可保留作默认，但不再依赖
#         self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
#         self.enable_fp16 = enable_fp16
#         self.cache_rays = cache_rays
#         self._ray_cache = {}
#
#     @staticmethod
#     def _key(H, W, K):
#         fx, fy, cx, cy = K[0,0].item(), K[1,1].item(), K[0,2].item(), K[1,2].item()
#         return (int(H), int(W), round(fx,3), round(fy,3), round(cx,1), round(cy,1))
#
#     @torch.no_grad()
#     def _get_unit_rays(self, H, W, K, device):
#         key = self._key(H, W, K)
#         if self.cache_rays and key in self._ray_cache:
#             return self._ray_cache[key].to(device, non_blocking=True)
#         fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
#         v = torch.arange(H, device=device).float()
#         u = torch.arange(W, device=device).float()
#         vv, uu = torch.meshgrid(v, u, indexing='ij')
#         x = (uu - cx) / fx
#         y = (vv - cy) / fy
#         z = torch.ones_like(x)
#         dirs = torch.stack([x, y, z], dim=-1)               # [H,W,3]
#         dirs = dirs / torch.clamp(dirs.norm(dim=-1, keepdim=True), min=1e-6)
#         if self.enable_fp16: dirs = dirs.half()
#         if self.cache_rays: self._ray_cache[key] = dirs.cpu()
#         return dirs
#
#     @torch.no_grad()
#     def project_tokens(self, pix_uv, depth_mu, K, T_cam2ego, H, W, Hb, Wb):
#         device = depth_mu.device
#         if pix_uv.numel() == 0:
#             mask = torch.zeros(0, dtype=torch.bool, device=device)
#             return mask, None
#
#         # —— dtype 对齐同你之前修过的 compute_dtype ——
#         compute_dtype = T_cam2ego.dtype
#
#         rays = self._get_unit_rays(H, W, K, device)
#         u = pix_uv[:, 0].round().long().clamp(0, W - 1)
#         v = pix_uv[:, 1].round().long().clamp(0, H - 1)
#         dir_cam = rays[v, u, :].to(dtype=compute_dtype)
#
#         R = T_cam2ego[:3, :3].to(device=device, dtype=compute_dtype)
#         t = T_cam2ego[:3, 3].to(device=device, dtype=compute_dtype)
#         depth_mu = depth_mu.to(dtype=compute_dtype)
#
#         dir_ego = (R @ dir_cam.transpose(0, 1)).transpose(0, 1)
#         cam_o = t.view(1, 3).expand_as(dir_ego)
#         xyz = cam_o + dir_ego * depth_mu.view(-1, 1)
#
#         x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
#         m = (x >= self.x_range[0]) & (x < self.x_range[1]) & \
#             (y >= self.y_range[0]) & (y < self.y_range[1]) & \
#             (z >= self.z_range[0]) & (z < self.z_range[1])
#         if m.sum() == 0:
#             return m, None
#
#         # ★ 动态步长：用“当前 Hb/Wb”算 dx/dy，保证索引与真实 BEV 对齐
#         dx = (self.x_range[1] - self.x_range[0]) / float(Wb)
#         dy = (self.y_range[1] - self.y_range[0]) / float(Hb)
#
#         j = ((x[m] - self.x_range[0]) / dx).long().clamp(0, Wb - 1)
#         i = ((y[m] - self.y_range[0]) / dy).long().clamp(0, Hb - 1)
#         ij = torch.stack([i, j], dim=-1)
#         return m, ij
# ---------- 门控 ----------
class RangeGate(nn.Module):
    def __init__(self, bev_h, bev_w, x_range, y_range,
                 alpha_min=0.2, alpha_max=1.0, r0=25.0, r1=80.0):
        super().__init__()
        self.x_range, self.y_range = x_range, y_range
        self.base_h, self.base_w = bev_h, bev_w
        yy, xx = torch.meshgrid(
            torch.linspace(y_range[0], y_range[1], bev_h),
            torch.linspace(x_range[0], x_range[1], bev_w),
            indexing='ij'
        )
        rr = torch.sqrt(xx**2 + yy**2)
        alpha = (rr - r0) / (r1 - r0 + 1e-6)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        alpha = alpha_min + (alpha_max - alpha_min) * alpha
        self.register_buffer('alpha_map', alpha.view(1,1,bev_h,bev_w))
        self._cached = {}  # {(H,W): tensor}

    def forward(self, H=None, W=None, dtype=None, device=None):
        ra = self.alpha_map
        if H is not None and W is not None and (H != ra.shape[-2] or W != ra.shape[-1]):
            key = (H, W)
            if key not in self._cached:
                self._cached[key] = F.interpolate(ra, size=(H, W), mode='bilinear', align_corners=False)
            ra = self._cached[key]
        if dtype is not None:
            ra = ra.to(dtype=dtype)
        if device is not None:
            ra = ra.to(device=device, non_blocking=True)
        return ra

class UncertaintyGate(nn.Module):
    def __init__(self, use_lidar_conf=True):
        super().__init__()
        in_c = 2 + (1 if use_lidar_conf else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_c, 32), nn.ReLU(True),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        self.use_lidar_conf = use_lidar_conf

    def forward(self, depth_logvar_cell, sem_conf_cell, lidar_conf_cell=None):
        depth_var = torch.exp(torch.clamp(depth_logvar_cell, -6, 6))
        depth_var = depth_var / (depth_var.max().detach() + 1e-6)
        feats = [depth_var.unsqueeze(-1), sem_conf_cell.unsqueeze(-1)]
        if self.use_lidar_conf and lidar_conf_cell is not None:
            feats.append(lidar_conf_cell.unsqueeze(-1))
        x = torch.cat(feats, dim=-1)
        return self.mlp(x).squeeze(-1)  # [N]

# ---------- 稀疏 Cross-Attn 融合 ----------
# class GatedCrossAttentionFuse(nn.Module):
#     def __init__(self, c_lidar: int, c_cam: int, heads=4, dim_head=32):
#         super().__init__()
#         self.q_proj = nn.Conv2d(c_lidar, heads*dim_head, 1)
#         self.k_proj = nn.Linear(c_cam, heads*dim_head, bias=False)
#         self.v_proj = nn.Linear(c_cam, heads*dim_head, bias=False)
#         self.out = nn.Conv2d(heads*dim_head, c_lidar, 1, bias=False)
#         self.h, self.d = heads, dim_head
#         # GPT 的偏移
#         # self.gamma = 0.05
#     def forward(self, lidar_bev, cam_bev_tokens, cam_bev_indices, gate_weights, range_alpha):
#         B, C, H, W = lidar_bev.shape
#         q = self.q_proj(lidar_bev).view(B, self.h, self.d, H, W)  # [B,h,d,H,W]
#         scale = 1.0 / math.sqrt(self.d)
#
#         # 用增量缓冲，避免在原 tensor/视图上原地写
#         delta = torch.zeros_like(lidar_bev)  # [B,C,H,W]
#
#         for b in range(B):
#             ind = cam_bev_indices[b]
#             if ind is None or ind.numel() == 0:
#                 continue
#
#             tok = cam_bev_tokens[b]
#             gw = gate_weights[b]
#             i = ind[:, 0].long().clamp(0, H - 1)
#             j = ind[:, 1].long().clamp(0, W - 1)
#             lin = i * W + j
#
#             q_b = q[b, :, :, i, j]  # [h,d,N]
#             tok = tok.to(q_b.dtype);
#             gw = gw.to(q_b.dtype)
#
#             k_b = self.k_proj(tok).to(q_b.dtype).view(-1, self.h, self.d).permute(1, 2, 0)  # [h,d,N]
#             v_b = self.v_proj(tok).to(q_b.dtype).view(-1, self.h, self.d).permute(1, 2, 0)  # [h,d,N]
#
#             attn = (q_b * k_b).sum(dim=1) * scale  # [h,N]
#             attn = F.softmax(attn, dim=-1) * gw.unsqueeze(0)  # [h,N]
#
#             weighted = attn.unsqueeze(1) * v_b  # [h,1,N]*[h,d,N] -> [h,d,N]
#             fused = weighted.reshape(self.h * self.d, -1)  # [h*d, N]
#
#             tmp_flat = torch.zeros(self.h * self.d, H * W, device=lidar_bev.device, dtype=fused.dtype)
#             tmp_flat.index_add_(1, lin, fused)  # 稀疏累加到展平平面
#             tmp = tmp_flat.view(self.h * self.d, H, W)
#
#             ra = range_alpha.to(dtype=tmp.dtype, device=tmp.device)
#             tmp = self.out(tmp.unsqueeze(0)) * ra  # [1, C, H, W]
#             #####  GPT加的减少偏移的
#             # ★ 关键：做零均值，去掉全局 DC 偏移
#             # tmp = tmp - tmp.mean(dim=(2, 3), keepdim=True)
#             #
#             # # ★ 残差缩放 + 距离门控
#             # tmp = tmp * self.gamma * range_alpha.to(dtype=tmp.dtype, device=tmp.device)
#
#             # 累加到“增量”，不动原始 lidar_bev
#             delta[b:b + 1] = delta[b:b + 1] + tmp
#
#         # 最终输出：原值 + 增量（非原地）
#         out = lidar_bev + delta
#
#         return out
class GatedCrossAttentionFuse(nn.Module):
    def __init__(self, c_lidar: int, c_cam: int, heads=4, dim_head=32):
        super().__init__()
        self.q_proj = nn.Conv2d(c_lidar, heads*dim_head, 1)
        self.k_proj = nn.Linear(c_cam, heads*dim_head, bias=False)
        self.v_proj = nn.Linear(c_cam, heads*dim_head, bias=False)
        self.out    = nn.Conv2d(heads*dim_head, c_lidar, 1, bias=False)  # bias=False
        self.h, self.d = heads, dim_head
        self.gamma = 0.08

    def forward(self, lidar_bev, cam_bev_tokens, cam_bev_indices, gate_weights, range_alpha):
        B, C, H, W = lidar_bev.shape
        q = self.q_proj(lidar_bev).view(B, self.h, self.d, H, W)
        scale = 1.0 / math.sqrt(self.d)

        # 把 1×1 卷积的权重摊平到矩阵，之后直接做 matmul
        # W_out: [C, h*d]
        W_out = self.out.weight.view(C, self.h * self.d)

        delta = torch.zeros_like(lidar_bev)

        for b in range(B):
            ind = cam_bev_indices[b]
            if ind is None or ind.numel() == 0:
                continue

            tok = cam_bev_tokens[b] # 有数值
            gw  = gate_weights[b]
            i = ind[:, 0].long().clamp(0, H-1)
            j = ind[:, 1].long().clamp(0, W-1)
            lin = i * W + j                          # [N]

            q_b = q[b, :, :, i, j]                   # [h,d,N]
            tok = tok.to(q_b.dtype);  gw = gw.to(q_b.dtype)

            k_b = self.k_proj(tok).to(q_b.dtype).view(-1, self.h, self.d).permute(1, 2, 0)  # [h,d,N]
            v_b = self.v_proj(tok).to(q_b.dtype).view(-1, self.h, self.d).permute(1, 2, 0)  # [h,d,N]

            attn = (q_b * k_b).sum(dim=1) * scale    # [h,N]
            attn = F.softmax(attn, dim=-1) * gw.unsqueeze(0)

            fused = (attn.unsqueeze(1) * v_b).reshape(self.h * self.d, -1)  # [(h*d), N]

            # ★ 先映射到 LiDAR 通道：out_tok = W_out @ fused  => [C, N]
            out_tok = W_out @ fused

            # ★ 直接在 C×(H*W) 平面累加（不再需要 (h*d)×(H*W) 中间图）
            delta_flat = torch.zeros(C, H*W, device=lidar_bev.device, dtype=out_tok.dtype)
            delta_flat.index_add_(1, lin, out_tok)   # [C, H*W]
            tmp = delta_flat.view(1, C, H, W)        # [1, C, H, W]

            # === 掩码零均值：只对命中过的格子去均值 ===
            hits = torch.bincount(lin, minlength=H * W).view(1, 1, H, W).to(tmp.dtype)
            mask = (hits > 0).to(tmp.dtype)
            eps = 1e-6
            mean_on_hits = (tmp * mask).sum(dim=(2, 3), keepdim=True) / (mask.sum(dim=(2, 3), keepdim=True) + eps)
            tmp = tmp - mean_on_hits * mask

            # 门控 + 残差缩放
            ra = range_alpha.to(dtype=tmp.dtype, device=tmp.device)   # [1,1,H,W] 或标量
            tmp = tmp * ra * float(self.gamma)

            delta[b:b+1] = delta[b:b+1] + tmp

        return lidar_bev + delta

# ---------- 顶层（注册） ----------
@MODELS.register_module()
class CamGuidedBEVWithFEMLite(BaseModule):
    """
    轻量相机->LiDAR BEV 增强：FEM(相机侧) + 小目标候选 + 不确定性 + Lite投影 + 门控Cross-Attn
    放置：ConvFuser 之前，对 lidar_bev 做增强。
    """
    def __init__(self,
                 bev_h, bev_w,
                 x_range, y_range, z_range,
                 c_lidar, c_cam,
                 use_lidar_conf=True,
                 topk_tokens=1200,
                 alpha_min=0.3, alpha_max=1.0, r0=25.0, r1=80.0,
                 use_fem_cam=True,
                 fem_scale_cam=0.1,
                 enable_fp16=True,
                 init_cfg=None,
                 # ↓↓↓ 新增：为了兼容 cfg
                 heads: int = 4,
                 dim_head: int = 32,
                 **kwargs
                 ):
        super().__init__(init_cfg=init_cfg)
        self.use_fem_cam = use_fem_cam
        self.topk = topk_tokens

        if use_fem_cam:
            self.fem_cam = FEMSmallSafe(in_planes=c_cam, out_planes=c_cam,
                                        stride=1, scale=fem_scale_cam,
                                        map_reduce=8, use_dilation=True,
                                        dilation_rates=(3,5), downsample_branch=None)
        self.candidate_head = SmallObjCandidateHead(c_cam)
        self.uncert_head = UncertaintyHead(c_cam)

        self.projector = LiteCamProjector(bev_h, bev_w, x_range, y_range, z_range,
                                          enable_fp16=enable_fp16, cache_rays=True)
        self.range_gate = RangeGate(bev_h, bev_w, x_range, y_range,
                                    alpha_min=alpha_min, alpha_max=alpha_max, r0=r0, r1=r1)
        self.uncert_gate = UncertaintyGate(use_lidar_conf=use_lidar_conf)
        self.cross_fuse = GatedCrossAttentionFuse(c_lidar=c_lidar, c_cam=c_cam, heads= heads, dim_head= dim_head)

    # def _sample_topk(self, cam_feat, heat, k):
    #     B, C, H, W = cam_feat.shape
    #     tokens, coords = [], []
    #     prob = torch.sigmoid(heat)
    #     for b in range(B):
    #         score = prob[b,0]
    #         kk = min(k, H*W)
    #         vals, idx = torch.topk(score.flatten(), k=kk, dim=-1)
    #         v = idx % W
    #         u = idx // W
    #         feat_b = cam_feat[b].permute(1,2,0).contiguous()  # [H,W,C]
    #         tok = feat_b[u, v, :]
    #         uv = torch.stack([v.float(), u.float()], dim=-1)  # (u,v)
    #         tokens.append(tok); coords.append(uv)
    #     return tokens, coords, H, W
    def _sample_topk(self, cam_feat, heat, k, grid=(8, 8), per_cell_cap=8):
        B, C, H, W = cam_feat.shape
        tokens, coords = [], []
        prob = torch.sigmoid(heat)

        gh, gw = grid
        hs, ws = H // gh, W // gw
        k_per = max(1, k // (gh * gw))

        for b in range(B):
            score = prob[b, 0]
            feat_b = cam_feat[b].permute(1, 2, 0).contiguous()  # [H,W,C]

            tok_list, uv_list = [], []
            for gy in range(gh):
                for gx in range(gw):
                    r0, r1 = gy * hs, min((gy + 1) * hs, H)
                    c0, c1 = gx * ws, min((gx + 1) * ws, W)
                    cell = score[r0:r1, c0:c1]
                    if cell.numel() == 0:
                        continue
                    kk = min(k_per, cell.numel())
                    vals, idx = torch.topk(cell.flatten(), k=kk, dim=-1)
                    v = idx % (c1 - c0)
                    u = idx // (c1 - c0)
                    u = u + r0
                    v = v + c0
                    tok_list.append(feat_b[u, v, :])
                    uv_list.append(torch.stack([v.float(), u.float()], dim=-1))

            if len(tok_list) == 0:
                tokens.append(torch.empty(0, C, device=cam_feat.device))
                coords.append(torch.empty(0, 2, device=cam_feat.device))
                continue

            tok = torch.cat(tok_list, 0)
            uv = torch.cat(uv_list, 0)

            # 全局再截断到总 topk
            if tok.shape[0] > k:
                svals = score[uv[:, 1].long(), uv[:, 0].long()]
                vals, keep_idx = torch.topk(svals, k=k, dim=0)
                tok = tok[keep_idx]
                uv = uv[keep_idx]

            tokens.append(tok)
            coords.append(uv)
        return tokens, coords, H, W
    def forward(self,
                lidar_bev: torch.Tensor,      # [B,C_l,Hb,Wb]
                cam_feat_p3: torch.Tensor,    # [B,C_cam,H,W] 选用 img_neck 的一层（建议P3）
                Ks: List[torch.Tensor],       # [3,3]，可传单个K或batch列表
                T_cam2egos: List[torch.Tensor],  # [4,4] * B
                lidar_occ_conf: Optional[torch.Tensor] = None  # [B,1,Hb,Wb]
                ):
        if self.use_fem_cam:
            cam_feat_p3 = self.fem_cam(cam_feat_p3)
            # self.last_fem_cam = cam_feat_p3
        heat = self.candidate_head(cam_feat_p3)                  # [B,1,H,W]
        # print('heat shape:', heat.shape)
        d_mu, d_logv, sem_conf = self.uncert_head(cam_feat_p3)   # [B,1,H,W] x3

        cam_tokens, pix_uv, H, W = self._sample_topk(cam_feat_p3, heat, self.topk)

        B, C_l, Hb, Wb = lidar_bev.shape #lidar_bev shape is 180 X 180
        # print('lidar_bev shape:', lidar_bev.shape)
        # print('shape of lidar_bev',Hb, Wb)
        device = lidar_bev.device
        if lidar_occ_conf is None:
            lidar_occ_conf = torch.ones(B,1,Hb,Wb, device=device)
        range_alpha = self.range_gate(Hb, Wb, dtype=lidar_bev.dtype, device=lidar_bev.device)
        #range_alpha = lidar_bev.new_tensor(0.85)  # 标量 0-D tensor
        cam_bev_tokens_b, cam_bev_indices_b, gate_weights_b = [], [], []
        for b in range(B):
            tok = cam_tokens[b]
            if tok.numel() == 0:
                cam_bev_tokens_b.append(torch.empty(0, tok.shape[-1], device=device))
                cam_bev_indices_b.append(None)
                gate_weights_b.append(torch.empty(0, device=device))
                continue

            uv = pix_uv[b].to(device)
            dmu_b, dlogv_b, sem_b = d_mu[b,0], d_logv[b,0], sem_conf[b,0]
            r_idx = uv[:,1].long().clamp(0, H-1); c_idx = uv[:,0].long().clamp(0, W-1)
            # r_idx = uv[:, 0].long().clamp(0, H - 1)  # row ← v
            # c_idx = uv[:, 1].long().clamp(0, W - 1)  # col ← u 这个是不正确的
            depth_mu_k     = dmu_b[r_idx, c_idx]
            depth_logvar_k = dlogv_b[r_idx, c_idx]
            sem_k          = sem_b[r_idx, c_idx]
            #depth_mu_k = (depth_mu_k * 15.0).clamp_min(1.0)  # 先试 scale=15，min=1m，ps:为了验证是不是深度的上下限的问题
            # 取完 depth_mu_k 后
            with torch.no_grad():
                med = depth_mu_k.median().clamp(min=1e-6)  # 当前帧中位数
                target = depth_mu_k.new_tensor(30.0)  # 目标中位数 ~ 30m
                scale = (target / med).clamp_(0.5, 60.0)  # 限幅，防止发散
            depth_mu_k = (depth_mu_k * scale).clamp_min(2.0)  # 2m 下限

            K = Ks[min(b, len(Ks)-1)].to(device)
            T = T_cam2egos[b].to(device)
            # projector 调用（已经传 Hb/Wb）
            mask, ij = self.projector.project_tokens(
                pix_uv=uv, depth_mu=depth_mu_k, K=K, T_cam2ego=T,
                H=H, W=W, Hb=Hb, Wb=Wb
            )

            # passed = 0 if (ij is None) else ij.shape[0]
            # total = tok.shape[0]
            # if ij is not None and passed > 0:
            #     uniq = torch.unique(ij, dim=0).shape[0]
            #     i_min, i_max = int(ij[:, 0].min()), int(ij[:, 0].max())
            #     j_min, j_max = int(ij[:, 1].min()), int(ij[:, 1].max())
            #     print(f'[proj] sample {b}: pass={passed}/{total}, '
            #           f'unique_cells={uniq}, i=[{i_min},{i_max}], j=[{j_min},{j_max}]')
            # else:
            #     print(f'[proj] sample {b}: pass=0/{total}, unique_cells=0')
            if mask.sum() == 0:
                cam_bev_tokens_b.append(torch.empty(0, tok.shape[-1], device=device))
                cam_bev_indices_b.append(None)
                gate_weights_b.append(torch.empty(0, device=device))
                continue
            tok_sel = tok[mask]
            i_sel, j_sel = ij[:,0], ij[:,1]
            lidar_conf_cell = lidar_occ_conf[b,0,i_sel,j_sel]
            g = self.uncert_gate(depth_logvar_cell=depth_logvar_k[mask],
                                 sem_conf_cell=sem_k[mask],
                                 lidar_conf_cell=lidar_conf_cell)
            cam_bev_tokens_b.append(tok_sel)
            cam_bev_indices_b.append(torch.stack([i_sel, j_sel], dim=-1))
            gate_weights_b.append(g)

        empty = torch.empty(0, 2, device=lidar_bev.device, dtype=torch.long)
        self.last_indices = [(inds if (inds is not None and inds.numel() > 0) else empty)
                             for inds in cam_bev_indices_b]
        self.last_gate_w = [(gw if gw.numel() > 0 else torch.empty(0, device=lidar_bev.device))
                            for gw in gate_weights_b]
        self.last_bev_hw = (Hb, Wb)

        if not hasattr(self, '_debug_once'):
            print('[cam_guided] tokens per sample:',
                  [int(v.shape[0]) for v in self.last_indices],
                  'BEV=', self.last_bev_hw)
            self._debug_once = True

        out = self.cross_fuse(lidar_bev, cam_bev_tokens_b, cam_bev_indices_b, gate_weights_b, range_alpha)
        # print('out shape:', out.shape)

        # self.last_heat = heat
        # self.last_indices = cam_bev_indices_b
        # self.last_gate_w = gate_weights_b
        # self.last_lidar_bev = lidar_bev #交互之前的lidar_bev
        # self.last_out = out #交互后的lidar_bev
        # self.last_bev_hw = (Hb, Wb)

        return out

@MODELS.register_module()
class LidarGuidedImgGating(nn.Module):
    """
    超轻量 LiDAR -> img_bev 门控增强：
    - 空间门控：gate = sigmoid(Conv1x1(lidar_bev_detached))
    - 可选乘上 range_alpha
    - 输出：img_bev * (1 + beta * gate)，保持残差稳定
    显存省：门控通路 no_grad + detach（只让梯度走 img_bev）
    """
    def __init__(self,
                 c_img: int = 64,
                 c_lidar: int = 256,
                 beta: float = 0.5,          # 门控强度
                 use_range_alpha: bool = True,
                 norm: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(c_lidar, 1, kernel_size=1, bias=True)
        self.beta = beta
        self.use_range_alpha = use_range_alpha
        self.norm = norm
        # 可选：给 img_bev 一个极轻的 1x1 适配（不一定需要）
        self.adapt_img = nn.Identity()

    def forward(self, img_bev: torch.Tensor,
                lidar_bev: torch.Tensor,
                range_alpha: torch.Tensor = None):
        """
        img_bev:   [B, 64, Hb, Wb]
        lidar_bev: [B, 256, Hb, Wb]
        """
        # ---- 1) 用 LiDAR 生成门控（no_grad，不进计算图）----
        with torch.no_grad():
            x = lidar_bev.detach()  # [B, C_l, H, W]
            gate = self.proj(x)  # [B, 1, H, W]

            if self.norm:
                # 零均值 + 标准差归一，避免 sigmoid 压成 0.5 常数
                mean = gate.mean(dim=(2, 3), keepdim=True)
                std = gate.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                gate = (gate - mean) / std

            gate = torch.sigmoid(gate)  # [B,1,H,W] ∈ (0,1)

            # 与 range_alpha 对齐（如果给了）
            if self.use_range_alpha and (range_alpha is not None):
                ra = range_alpha
                if ra.shape[-2:] != gate.shape[-2:]:
                    ra = F.interpolate(ra, size=gate.shape[-2:],
                                       mode='bilinear', align_corners=False)
                gate = gate * ra.to(dtype=gate.dtype, device=gate.device)

            # 再做一次轻量 min-max，拉开对比度（不会增参）
            gmin = gate.amin(dim=(2, 3), keepdim=True)
            gmax = gate.amax(dim=(2, 3), keepdim=True)
            gate = (gate - gmin) / (gmax - gmin + 1e-6)

        # ---- 2) 残差增强：y = img + beta*(img*gate)（更省显存的 addcmul）----
        gate = gate.to(dtype=img_bev.dtype)  # 与特征 dtype 对齐（fp16 更省）
        # if gate.shape[-2:] != img_bev.shape[-2:]:
        #     # 让 gate 跟 img_bev 对齐
        #     gate = F.interpolate(gate, size=img_bev.shape[-2:], mode='bilinear', align_corners=False)
        y = torch.addcmul(img_bev, img_bev, gate, value=self.beta)

        # print(y.shape)

        # ---- 3) 用于可视化：保存门控图而不是输出特征 ----
        # self.last_gate = y  # [B,1,H,W]

        return y

@MODELS.register_module()
class ConvFuser(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(
                sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # ref = inputs[0]
        # inputs = [_resize_like(t, ref) for t in inputs]
        return super().forward(torch.cat(inputs, dim=1))
############### 多尺度BEV特征空间融合 ################
@MODELS.register_module()
class MSSA_Lite(nn.Module):
    """
    Lightweight Multi-Scale Spatial Self-Attention (单帧空间增强)
    - 多尺度depthwise卷积 + 空间注意力
    - 提升单帧BEV特征的空间信息建模能力
    """
    def __init__(self, channels=256, reduction=2):
        super().__init__()
        sizes = [7, 15]

        # 多尺度depthwise卷积
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (1, s), padding=(0, s//2), groups=channels),
                nn.Conv2d(channels, channels, (s, 1), padding=(s//2, 0), groups=channels)
            ) for s in sizes
        ])

        # 轻量空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )

        # 输出卷积
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 多尺度特征聚合
        multi_scale = sum(branch(x) for branch in self.branches) / len(self.branches)
        # 空间注意力
        attn = self.spatial_attn(multi_scale)
        out = x + multi_scale * attn
        # print('after mssw:',out.shape)
        return out
        # y = self.out_conv(out)
        # return y
############# 轻量化历史帧对齐网络  ##############
class LightHANet(nn.Module):
    """
    轻量历史对齐模块（Depthwise DeformConv + Channel Attention）
    """
    def __init__(self, channels=256, kernel_size=5, reduction=8):
        super().__init__()
        padding = kernel_size // 2
        mid_ch = channels // reduction

        self.offset = nn.Conv2d(channels * 2, 2 * kernel_size * kernel_size, 1)
        self.deform = DeformConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,  # depthwise
            bias=False
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, cur, hist):
        offset = self.offset(torch.cat([cur, hist], dim=1))
        aligned = self.deform(hist, offset)
        w = self.channel_attn(cur - aligned)
        fused = cur + aligned * w
        return self.out(fused)

@MODELS.register_module()
class LightAlignedMemoryBank(nn.Module):
    """
    多帧时序对齐融合模块
    - FIFO缓存历史BEV特征
    - Depthwise Deform对齐 + 通道加权融合
    """
    def __init__(self,
                 bev_channels=256,
                 max_length=2,
                 fuse_type='conv',
                 kernel_size=5):
        super().__init__()
        self.max_length = max_length
        self.memory = deque(maxlen=max_length)
        self.ha_net = LightHANet(bev_channels, kernel_size)
        self.fuse_type = fuse_type

        if fuse_type == 'conv':
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(inplace=True)
            )
    # -----------------------------------------------------
    # 手动重置记忆（每个 epoch 开始 / resume 时调用）
    # -----------------------------------------------------
    def reset(self):
        self.memory.clear()
    # -----------------------------------------------------
    # 模式切换时自动清空（保证 test 不残留历史）
    # -----------------------------------------------------
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.reset()
        else:
            # eval 时也清一次，防止状态残留
            self.reset()
        return self
    # -----------------------------------------------------
    # 前向传播
    # -----------------------------------------------------
    def forward(self, cur_bev_feat):
        # ========================
        # Eval 阶段：单帧模式
        # ========================
        # if not self.training:
        #     # 不使用任何记忆，只返回当前特征
        #     return cur_bev_feat  #eval的部分 我还是保存当前的特征,不做任何的操作
        # ========================
        # Train 阶段：多帧融合
        # ========================
        if len(self.memory) == 0:
            self.memory.append(cur_bev_feat.detach())
            return cur_bev_feat

        aligned_sum, weight_sum = 0, 0
        for hist in list(self.memory):
            # --- 尺寸防护 ---
            if hist.ndim == 5:
                hist = hist[-1]
            if hist.shape[-2:] != cur_bev_feat.shape[-2:]:
                hist = F.interpolate(hist, size=cur_bev_feat.shape[-2:], mode='bilinear', align_corners=False)
            if hist.shape[0] != cur_bev_feat.shape[0]:
                hist = hist[:cur_bev_feat.shape[0]]

            # --- 对齐 ---
            aligned = self.ha_net(cur_bev_feat, hist)

            # --- 通道加权 ---
            diff = F.adaptive_avg_pool2d(torch.abs(cur_bev_feat - hist), 1)
            w = torch.sigmoid(-diff)

            aligned_sum += aligned * w
            weight_sum += w

        fused = aligned_sum / (weight_sum + 1e-6)

        # --- 平滑融合 ---
        if self.fuse_type == 'conv':
            fused = self.fuse_conv(fused)

        # 更新记忆
        self.memory.append(cur_bev_feat.detach())

        return fused #去掉单帧之后，结果直接上涨了

@MODELS.register_module()
class TransFusionHead(nn.Module):

    def __init__(
        self,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        decoder_layer=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',
        # loss
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean'),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super(TransFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(decoder_layer))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def forward_single(self, inputs, metas):
        """Forward function for CenterPoint.
        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)
        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        fusion_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        fusion_feat_flatten = fusion_feat.view(batch_size,
                                               fusion_feat.shape[1],
                                               -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        #################################
        # query initialization
        #################################
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(
                heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(
                heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, fusion_feat_flatten.shape[1], -1),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(
            top_proposals_class,
            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        #################################
        # transformer decoder layer (Fusion feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](
                query_feat,
                key=fusion_feat_flatten,
                query_pos=query_pos,
                key_pos=bev_pos)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(
                0, 2, 1)
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:,
                                      None, :].expand(-1, self.num_classes,
                                                      -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dicts[0]['dense_heatmap'] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in [
                    'dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score'
            ]:
                new_res[key] = torch.cat(
                    [ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second
            index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats, [metas])
        assert len(res) == 1, 'only support one level features.'
        return res

    def predict(self, batch_feats, batch_input_metas):
        preds_dicts = self(batch_feats, batch_input_metas)
        res = self.predict_by_feat(preds_dicts, batch_input_metas)
        return res

    def predict_by_feat(self,
                        preds_dicts,
                        metas,
                        img=None,
                        rescale=False,
                        for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer
            & each batch.
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid()) # noqa: E501
            one_hot = F.one_hot(
                self.query_labels,
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    ))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.
                                    test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)

                temp_instances = InstanceData()
                temp_instances.bboxes_3d = metas[0]['box_type_3d'](
                    ret['bboxes'], box_dim=ret['bboxes'].shape[-1])
                temp_instances.scores_3d = ret['scores']
                temp_instances.labels_3d = ret['labels'].int()

                ret_layer.append(temp_instances)

            rets.append(ret_layer)
        assert len(
            rets
        ) == 1, f'only support one layer now, but get {len(rets)} layers'

        return rets[0]

    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: List[dict]):
        """Generate training targets.
        Args:
            batch_gt_instances_3d (List[InstanceData]):
            preds_dict (list[dict]): The prediction results. The index of the
                list is the index of layers. The inner dict contains
                predictions of one mini-batch:
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: The original heatmap before fed into transformer
                    decoder, with shape (bs, 10, h, w)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                    [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                preds = []
                for i in range(self.num_decoder_layers):
                    pred_one_layer = preds_dict[i][key][batch_idx:batch_idx +
                                                        1]
                    preds.append(pred_one_layer)
                pred_dict[key] = torch.cat(preds)
            list_of_pred_dict.append(pred_dict)

        assert len(batch_gt_instances_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
            np.arange(len(batch_gt_instances_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """
        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals *
                                                idx_layer:self.num_proposals *
                                                (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals *
                                idx_layer:self.num_proposals *
                                (idx_layer + 1), ]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    gt_labels_3d,
                    score_layer,
                    self.train_cfg,
                )
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(
                    bboxes_tensor_layer,
                    gt_bboxes_tensor,
                    None,
                    gt_labels_3d,
                    self.query_labels[batch_idx],
                )
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )

        # 2. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   pred_instances,
                                                   gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # 3. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']
                            )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1],
                                         feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
                # NOTE: fix
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                      center_int[[1, 0]], radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    def loss(self, batch_feats, batch_data_samples):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        batch_input_metas, batch_gt_instances_3d = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
        preds_dicts = self(batch_feats, batch_input_metas)
        loss = self.loss_by_feat(preds_dicts, batch_gt_instances_3d)

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(batch_gt_instances_3d, preds_dicts[0])
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict['dense_heatmap']).float(),
            heatmap.float(),
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(
                self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                    idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer *
                                  self.num_proposals:(idx_layer + 1) *
                                  self.num_proposals, ].reshape(-1)
            layer_label_weights = label_weights[
                ..., idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals, ].reshape(-1)
            layer_score = preds_dict['heatmap'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(
                -1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score.float(),
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_center = preds_dict['center'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_height = preds_dict['height'][..., idx_layer *
                                                self.num_proposals:(idx_layer +
                                                                    1) *
                                                self.num_proposals, ]
            layer_rot = preds_dict['rot'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            layer_dim = preds_dict['dim'][..., idx_layer *
                                          self.num_proposals:(idx_layer + 1) *
                                          self.num_proposals, ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, ]
                preds = torch.cat([
                    layer_center, layer_height, layer_dim, layer_rot, layer_vel
                ],
                                  dim=1).permute(
                                      0, 2,
                                      1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(  # noqa: E501
                code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer *
                                              self.num_proposals:(idx_layer +
                                                                  1) *
                                              self.num_proposals, :, ]
            layer_loss_bbox = self.loss_bbox(
                preds,
                layer_bbox_targets,
                layer_reg_weights,
                avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict
