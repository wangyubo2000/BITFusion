_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    # ===== 单帧空间增强 =====
    mssa_plugin=dict(
        type='MSSA_Lite',
        channels=256,
        reduction=2
    ), # 后期也得
    # ===== 时序对齐融合 =====
    memory_bank=dict(
        type='LightAlignedMemoryBank',
        bev_channels=256,
        max_length=1, # 后期可以修改
        fuse_type='conv',
        kernel_size=3 # 后期可以修改
    )
)

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,   #num = 9
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # Actually, 'GridMask' is not used here
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5, # orginal = 9
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size=2,  # [MOD-BS2] 单卡上把 batch 提到 2
    dataset=dict(dataset=dict(pipeline=train_pipeline, modality=input_modality))
)
val_dataloader = dict(
    batch_size=1,  # [MOD-BS2] 验证时设 1，避免验证阶段显存峰值
    dataset=dict(pipeline=test_pipeline, modality=input_modality)
)
test_dataloader = val_dataloader
param_scheduler = [
    # ---- Warmup ----
    dict(
        type='LinearLR',
        start_factor=0.2,       # 起步略高，减少过快饱和
        by_epoch=False,
        begin=0,
        end=800),               # 略缩短 warmup，加快进入主训练

    # ---- Cosine 学习率 ----
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=6,
        end=6,
        by_epoch=True,
        eta_min_ratio=3e-3,     # 尾部稍高，防止过早过拟合
        convert_to_iter_based=True),

    # ---- Cosine 动量前半段：下降到 0.85 ----
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85,
        begin=0,
        end=2.4,
        by_epoch=True,
        convert_to_iter_based=True),

    # ---- Cosine 动量后半段：回升到 0.97 ----
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.97,
        begin=2.4,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.01),
#     loss_scale='dynamic',  # ★ 自动调整放大比例，防止梯度上溢或下溢
#     clip_grad=dict(max_norm=35, norm_type=2))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=3e-5,                # ↓ 降低初始 lr，收敛更平缓
        weight_decay=0.05,      # ↑ 提升正则，缓解过拟合
        eps=1e-8),              # ↑ 更稳的 fp16 精度
    loss_scale='dynamic',   # AMP 溢出自动跳步
    clip_grad=dict(max_norm=35, norm_type=2),
    dtype='float16'         # AMP 启用
)
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=2)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
custom_hooks = _base_.custom_hooks + [
    dict(type='EmptyCacheHook', after_iter=True)
]
# del _base_.custom_hooks