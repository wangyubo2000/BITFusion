# modify from https://github.com/mit-han-lab/bevfusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class GeneralizedLSSFPN(BaseModule): #多尺度采样+特征的融合

    def __init__(
            self,
            in_channels,  # 输入通道
            out_channels, # 输出通道
            num_outs,   # 有几个尺度层
            start_level=0, #控制从第几个层开始
            end_level=-1,
            no_norm_on_lateral=False,
            conv_cfg=None,
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(mode='bilinear', align_corners=True),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1   # 这里就是3-1=2
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i] +
                (in_channels[i + 1] if i == self.backbone_end_level -
                 1 else out_channels),
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))]

        # build top-down path
        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            )
            laterals[i] = torch.cat([laterals[i], x], dim=1)
            laterals[i] = self.lateral_convs[i](laterals[i])
            laterals[i] = self.fpn_convs[i](laterals[i])

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)]
        return tuple(outs)

class _BasicConv(nn.Module):
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


class _FEMSmallSafe(nn.Module):
    """
    小目标友好的 FEM：默认 stride=1；非对称核+空洞扩受野；不暗降分辨率。
    """
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8,
                 use_dilation=True, dilation_rates=(3, 5), downsample_branch=None):
        super().__init__()
        self.scale = scale
        inter = max(1, in_planes // map_reduce)
        self.b0 = nn.Sequential(
            _BasicConv(in_planes, 2*inter, 1, 1, 0),
            _BasicConv(2*inter, 2*inter, 3, 1, 1, relu=False),
        )
        s1 = stride if (downsample_branch == 1) else 1
        self.b1 = nn.Sequential(
            _BasicConv(in_planes, inter, 1, 1, 0),
            _BasicConv(inter, (inter//2)*3, (1,3), s1, (0,1)),
            _BasicConv((inter//2)*3, 2*inter, (3,1), 1, (1,0)),
            _BasicConv(2*inter, 2*inter, 3, 1,
                      padding=(dilation_rates[0] if use_dilation else 1),
                      dilation=(dilation_rates[0] if use_dilation else 1),
                      relu=False)
        )
        s2 = stride if (downsample_branch == 2) else 1
        self.b2 = nn.Sequential(
            _BasicConv(in_planes, inter, 1, 1, 0),
            _BasicConv(inter, (inter//2)*3, (3,1), s2, (1,0)),
            _BasicConv((inter//2)*3, 2*inter, (1,3), 1, (0,1)),
            _BasicConv(2*inter, 2*inter, 3, 1,
                      padding=(dilation_rates[1] if use_dilation else 1),
                      dilation=(dilation_rates[1] if use_dilation else 1),
                      relu=False)
        )
        self.merge = _BasicConv(6*inter, out_planes, 1, 1, 0, relu=False)
        self.short = _BasicConv(in_planes, out_planes, 1,
                               stride=(stride if (downsample_branch is not None) else 1),
                               relu=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.b0(x); x1 = self.b1(x); x2 = self.b2(x)
        out = self.merge(torch.cat([x0, x1, x2], dim=1))
        out = out * self.scale + self.short(x)
        return self.relu(out)


class _SmallObjCandidateHead(nn.Module):
    """小目标候选热图 head：输入 C=FPN 通道（如 256），输出 1 通道热度图。"""
    def __init__(self, in_channels, mid=128):
        super().__init__()
        self.net = nn.Sequential(
            _BasicConv(in_channels, mid, 3, 1, 1),
            _BasicConv(mid, mid, 3, 1, 1),
            nn.Conv2d(mid, 1, 1)
        )
    def forward(self, x):
        return self.net(x)  # [B,1,H,W] logits


class _UncertaintyHead(nn.Module):
    """不确定性：深度均值/对数方差 + 语义置信"""
    def __init__(self, in_channels, mid=128):
        super().__init__()
        self.shared = nn.Sequential(
            _BasicConv(in_channels, mid, 3, 1, 1),
            _BasicConv(mid, mid, 3, 1, 1)
        )
        self.depth_mu = nn.Conv2d(mid, 1, 1)
        self.depth_logvar = nn.Conv2d(mid, 1, 1)
        self.sem_logit = nn.Conv2d(mid, 1, 1)
    def forward(self, x):
        h = self.shared(x)
        d_mu = F.softplus(self.depth_mu(h))    # >0
        d_logv = self.depth_logvar(h)
        sem = torch.sigmoid(self.sem_logit(h)) # 0..1
        return d_mu, d_logv, sem

@MODELS.register_module()
class GeneralizedLSSFPNWithFEM(GeneralizedLSSFPN):
    """
    兼容原 GeneralizedLSSFPN 的派生类：
    - forward 返回仍然是 tuple(outs)
    - 同时在 outs[0](P3) 上做 FEM + 候选 + 不确定性，并把结果缓存到属性上
      供后续 BEVFusion Detector 里的相机→LiDAR 插件使用。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN2d'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(mode='bilinear', align_corners=True),
                 # ---- 新增开关与参数 ----
                 use_fem_on_p3: bool = True,
                 fem_scale: float = 0.1,
                 fem_map_reduce: int = 8,
                 cand_mid_channels: int = 128,
                 uncert_mid_channels: int = 128):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         num_outs=num_outs,
                         start_level=start_level,
                         end_level=end_level,
                         no_norm_on_lateral=no_norm_on_lateral,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         upsample_cfg=upsample_cfg)
        self.use_fem_on_p3 = use_fem_on_p3

        # 运行期缓存（供外部读取，不影响 forward 返回）
        self.last_p3_enh = None
        self.last_heat = None
        self.last_depth_mu = None
        self.last_depth_logvar = None
        self.last_sem_conf = None

        if self.use_fem_on_p3:
            # P3 的通道就是 FPN out_channels（你 cfg 里是 256）
            c = out_channels
            self.fem_p3 = _FEMSmallSafe(c, c, stride=1, scale=fem_scale, map_reduce=fem_map_reduce,
                                        use_dilation=True, dilation_rates=(3,5), downsample_branch=None)
            self.cand_head = _SmallObjCandidateHead(c, mid=cand_mid_channels)
            self.uncert_head = _UncertaintyHead(c, mid=uncert_mid_channels)

    def forward(self, inputs):
        """
        保持与你原始 GeneralizedLSSFPN 完全一致的行为：返回 tuple(outs)
        但同时把 P3 的增强与热图/不确定性缓存到属性上。
        """
        outs = super().forward(inputs)   # tuple of [outs[0]=P3, outs[1]=P4, outs[2]=P5]（按你 num_outs）
        if self.use_fem_on_p3 and len(outs) > 0:
            p3 = outs[0]                                # [B, C=out_channels, H, W]
            p3_enh = self.fem_p3(p3)                    # FEM 增强
            heat = self.cand_head(p3_enh)               # [B,1,H,W] logits
            d_mu, d_logv, sem = self.uncert_head(p3_enh)

            # 缓存到模块属性，供 Detector 使用
            self.last_p3_enh = p3_enh
            self.last_heat = heat
            self.last_depth_mu = d_mu
            self.last_depth_logvar = d_logv
            self.last_sem_conf = sem

            # 注意：我们**不改 outs**，保持兼容（view_transform 继续吃原 P3）
            # 如果你想让 view_transform 也用增强后的 P3，可以在这里替换 outs[0]=p3_enh
            # outs = (p3_enh, ) + outs[1:]
        else:
            # 没开 FEM 就把缓存清空
            self.last_p3_enh = None
            self.last_heat = None
            self.last_depth_mu = None
            self.last_depth_logvar = None
            self.last_sem_conf = None
        return outs
