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
        max_length=2, # 后期可以修改
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

tta_model = dict(
    type='Det3DTTAModel',
    tta_cfg=dict(
        use_rotate_nms=True,   # 保留旋转NMS
        nms_thr=0.05,
        max_num=500,
        score_thr=0.1  # 先高一点，压低分噪框
    )
)
tta_pipeline = [
    dict(
        type='MultiScaleFlipAug3D',
        flip=True,
        img_scale=(704, 256),      # 与 ImageAug3D.final_dim 对应（W,H）or 保持一致即可
        pts_scale_ratio=[1.0],
        pcd_horizontal_flip=True,
        pcd_vertical_flip=False,
        transforms=[
            dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', backend_args=backend_args),
            dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
            dict(type='LoadPointsFromMultiSweeps', sweeps_num=5, load_dim=5, use_dim=5,
                 pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
            dict(type='ImageAug3D', final_dim=[256, 704], resize_lim=[0.48, 0.48], bot_pct_lim=[0.0, 0.0],
                 rot_lim=[0.0, 0.0], rand_flip=False, is_train=False),
# ★★ 关键：真正执行翻转，同时同步 2D（按照 MultiScaleFlipAug3D 写入的 flags 来决定是否翻）★★
            dict(type='RandomFlip3D',
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,  # 由 results['pcd_horizontal_flip'] 控制
                 flip_ratio_bev_vertical=0.0),   # 由 results['pcd_vertical_flip'] 控制
            dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
            dict(
                type='Pack3DDetInputs',
                keys=['img', 'points'],
                meta_keys=[
                    'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
                    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'box_mode_3d',
                    'sample_idx', 'lidar_path', 'img_path', 'num_pts_feats',
                    'pcd_scale_factor', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'pcd_rotation'
                ])
        ]
    )
]
# tta_pipeline = [
#     dict(
#         type='MultiScaleFlipAug3D',
#         flip=True,                     # ✅ 保留图像flip
#         img_scale=(800, 448),
#         pts_scale_ratio=[1.0, 1.05],   # ✅ 缩减scale组合
#         pcd_horizontal_flip=True,      # ✅ 只保留水平翻转
#         pcd_vertical_flip=False,       # ❌ 去掉垂直翻转
#         transforms=[
#             dict(
#                 type='BEVLoadMultiViewImageFromFiles',
#                 to_float32=True,
#                 color_type='color',
#                 backend_args=backend_args
#             ),
#             dict(
#                 type='LoadPointsFromFile',
#                 coord_type='LIDAR',
#                 load_dim=5,
#                 use_dim=5,
#                 backend_args=backend_args
#             ),
#             dict(
#                 type='LoadPointsFromMultiSweeps',
#                 sweeps_num=5,
#                 load_dim=5,
#                 use_dim=5,
#                 pad_empty_sweeps=True,
#                 remove_close=True,
#                 backend_args=backend_args
#             ),
#             dict(
#                 type='ImageAug3D',
#                 final_dim=[256, 704],
#                 resize_lim=[0.48, 0.52],  # ✅ 缩小扰动范围
#                 bot_pct_lim=[0.0, 0.0],
#                 rot_lim=[-0.03, 0.03],    # ✅ 小范围旋转增强
#                 rand_flip=False,          # ❌ 避免与flip重复
#                 is_train=False
#             ),
#             dict(
#                 type='PointsRangeFilter',
#                 point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
#             ),
#             dict(
#                 type='Pack3DDetInputs',
#                 keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
#                 meta_keys=[
#                     'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
#                     'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
#                     'lidar_path', 'img_path', 'num_pts_feats',
#                     'pcd_scale_factor', 'pcd_horizontal_flip', 'pcd_vertical_flip'
#                 ]
#             )
#         ])
# ]


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
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=6,
        end=6,
        by_epoch=True,
        eta_min_ratio=1e-3,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.9,
        begin=0,
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
        lr=5e-5,            # ↓ 降低学习率
        weight_decay=0.01,
        eps=1e-6            # ↑ 增强数值稳
    ),
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