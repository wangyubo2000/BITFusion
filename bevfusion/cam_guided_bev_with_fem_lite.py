# cam_guided_bev_with_fem_lite.py
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

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
                 use_dilation=True, dilation_rates=(3,5), downsample_branch=None):
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
        return self.relu(out)


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
        return self.conv(x)

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
        d_mu = F.softplus(self.depth_mu(h))
        d_logv = self.depth_logvar(h)
        sem = torch.sigmoid(self.sem_logit(h))
        return d_mu, d_logv, sem


# ---------- 轻量投影器（稀疏 token，近邻射线） ----------
class LiteCamProjector(nn.Module):
    def __init__(self, bev_h, bev_w, x_range, y_range, z_range,
                 enable_fp16=True, cache_rays=True):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
        self.dx = (x_range[1] - x_range[0]) / bev_w
        self.dy = (y_range[1] - y_range[0]) / bev_h
        self.enable_fp16 = enable_fp16
        self.cache_rays = cache_rays
        self._ray_cache = {}

    @staticmethod
    def _key(H, W, K):
        fx, fy, cx, cy = K[0,0].item(), K[1,1].item(), K[0,2].item(), K[1,2].item()
        return (int(H), int(W), round(fx,3), round(fy,3), round(cx,1), round(cy,1))

    @torch.no_grad()
    def _get_unit_rays(self, H, W, K, device):
        key = self._key(H, W, K)
        if self.cache_rays and key in self._ray_cache:
            return self._ray_cache[key].to(device, non_blocking=True)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        v = torch.arange(H, device=device).float()
        u = torch.arange(W, device=device).float()
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        x = (uu - cx) / fx
        y = (vv - cy) / fy
        z = torch.ones_like(x)
        dirs = torch.stack([x, y, z], dim=-1)               # [H,W,3]
        dirs = dirs / torch.clamp(dirs.norm(dim=-1, keepdim=True), min=1e-6)
        if self.enable_fp16: dirs = dirs.half()
        if self.cache_rays: self._ray_cache[key] = dirs.cpu()
        return dirs

    @torch.no_grad()
    def project_tokens(self,
                       pix_uv: torch.Tensor,       # [N,2] (u,v)
                       depth_mu: torch.Tensor,     # [N]
                       K: torch.Tensor,            # [3,3]
                       T_cam2ego: torch.Tensor,    # [4,4]
                       H: int, W: int,             # 相机特征图尺寸
                       Hb: int, Wb: int            # BEV 尺寸
                       ):
        device = depth_mu.device
        if pix_uv.numel() == 0:
            mask = torch.zeros(0, dtype=torch.bool, device=device)
            return mask, None
        rays = self._get_unit_rays(H, W, K, device)         # [H,W,3]
        u = pix_uv[:,0].round().long().clamp(0, W-1)
        v = pix_uv[:,1].round().long().clamp(0, H-1)
        dir_cam = rays[v, u, :]                              # [N,3] fp16/32

        R = T_cam2ego[:3,:3].to(device)
        t = T_cam2ego[:3, 3].to(device)
        dir_ego = (R @ dir_cam.transpose(0,1)).transpose(0,1)  # [N,3]
        cam_o = t.view(1,3).expand_as(dir_ego)
        xyz = cam_o + dir_ego * depth_mu.view(-1,1)

        x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
        m = (x>=self.x_range[0]) & (x<self.x_range[1]) & \
            (y>=self.y_range[0]) & (y<self.y_range[1]) & \
            (z>=self.z_range[0]) & (z<self.z_range[1])
        if m.sum() == 0:
            return m, None
        j = ((x[m] - self.x_range[0]) / self.dx).long().clamp(0, Wb-1)
        i = ((y[m] - self.y_range[0]) / self.dy).long().clamp(0, Hb-1)
        ij = torch.stack([i, j], dim=-1)                    # [N_in,2]
        return m, ij


# ---------- 门控 ----------
class RangeGate(nn.Module):
    def __init__(self, bev_h, bev_w, x_range, y_range, alpha_min=0.2, alpha_max=1.0, r0=25.0, r1=80.0):
        super().__init__()
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

    def forward(self): return self.alpha_map

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
class GatedCrossAttentionFuse(nn.Module):
    def __init__(self, c_lidar: int, c_cam: int, heads=4, dim_head=32):
        super().__init__()
        self.q_proj = nn.Conv2d(c_lidar, heads*dim_head, 1)
        self.k_proj = nn.Linear(c_cam, heads*dim_head, bias=False)
        self.v_proj = nn.Linear(c_cam, heads*dim_head, bias=False)
        self.out = nn.Conv2d(heads*dim_head, c_lidar, 1)
        self.h, self.d = heads, dim_head

    def forward(self, lidar_bev, cam_bev_tokens, cam_bev_indices, gate_weights, range_alpha):
        B, C, H, W = lidar_bev.shape
        q = self.q_proj(lidar_bev).view(B, self.h, self.d, H, W)  # [B,h,d,H,W]
        out = lidar_bev
        scale = 1.0 / math.sqrt(self.d)

        for b in range(B):
            ind = cam_bev_indices[b]
            if ind is None or ind.numel() == 0: continue
            tok = cam_bev_tokens[b]      # [N, C_cam]
            gw  = gate_weights[b]        # [N]
            i = ind[:,0].long().clamp(0,H-1)
            j = ind[:,1].long().clamp(0,W-1)

            q_b = q[b, :, :, i, j]       # [h,d,N]
            k_b = self.k_proj(tok).view(-1, self.h, self.d).permute(1,2,0)  # [h,d,N]
            v_b = self.v_proj(tok).view(-1, self.h, self.d).permute(1,2,0)  # [h,d,N]

            attn = (q_b * k_b).sum(dim=1) * scale
            attn = F.softmax(attn, dim=-1) * gw.unsqueeze(0)
            fused = torch.einsum('hn,hdn->hdn', attn, v_b).sum(dim=0)  # [d,N]

            tmp = torch.zeros(self.h*self.d, H, W, device=out.device)
            # 稀疏写回（可改为 index_put_ / scatter_add_ 做平均）
            for n in range(fused.shape[1]):
                tmp[:, i[n], j[n]] += fused[:, n]
            tmp = self.out(tmp.unsqueeze(0)) * range_alpha
            out[b:b+1] = out[b:b+1] + tmp
        return out


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
                 init_cfg=None):
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
        self.cross_fuse = GatedCrossAttentionFuse(c_lidar=c_lidar, c_cam=c_cam, heads=4, dim_head=32)

    def _sample_topk(self, cam_feat, heat, k):
        B, C, H, W = cam_feat.shape
        tokens, coords = [], []
        prob = torch.sigmoid(heat)
        for b in range(B):
            score = prob[b,0]
            kk = min(k, H*W)
            vals, idx = torch.topk(score.flatten(), k=kk, dim=-1)
            v = idx % W
            u = idx // W
            feat_b = cam_feat[b].permute(1,2,0).contiguous()  # [H,W,C]
            tok = feat_b[u, v, :]
            uv = torch.stack([v.float(), u.float()], dim=-1)  # (u,v)
            tokens.append(tok); coords.append(uv)
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

        heat = self.candidate_head(cam_feat_p3)                  # [B,1,H,W]
        d_mu, d_logv, sem_conf = self.uncert_head(cam_feat_p3)   # [B,1,H,W] x3
        cam_tokens, pix_uv, H, W = self._sample_topk(cam_feat_p3, heat, self.topk)

        B, C_l, Hb, Wb = lidar_bev.shape
        device = lidar_bev.device
        if lidar_occ_conf is None:
            lidar_occ_conf = torch.ones(B,1,Hb,Wb, device=device)
        range_alpha = self.range_gate()

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
            depth_mu_k     = dmu_b[r_idx, c_idx]
            depth_logvar_k = dlogv_b[r_idx, c_idx]
            sem_k          = sem_b[r_idx, c_idx]

            K = Ks[min(b, len(Ks)-1)].to(device)
            T = T_cam2egos[b].to(device)
            mask, ij = self.projector.project_tokens(
                pix_uv=uv, depth_mu=depth_mu_k, K=K, T_cam2ego=T,
                H=H, W=W, Hb=Hb, Wb=Wb
            )
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
        # 供 BEVFusion.loss() 使用的缓存（仅当前 batch） -> 在对齐的时候也应该有一个约束
        empty = torch.empty(0, 2, device=lidar_bev.device, dtype=torch.long)
        self.last_indices = [(inds if (inds is not None and inds.numel() > 0) else empty)
                             for inds in cam_bev_indices_b]
        self.last_gate_w = [(gw if gw.numel() > 0 else torch.empty(0, device=lidar_bev.device))
                            for gw in gate_weights_b]
        self.last_bev_hw = (Hb, Wb)

        out = self.cross_fuse(lidar_bev, cam_bev_tokens_b, cam_bev_indices_b, gate_weights_b, range_alpha)
        return out
