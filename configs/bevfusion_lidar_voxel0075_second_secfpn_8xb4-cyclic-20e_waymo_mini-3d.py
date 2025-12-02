_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion'], allow_failed_imports=False)
# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']
grid_size = [1504,1504, 41] #这个需要，后面已经修改了
metainfo = dict(classes=class_names) #len=3
dataset_type = 'WaymoDataset' #修改成waymo
# data_root = 'data/nuscenes/'
data_root= '/home/wangyubo/data/Waymo_mini/kitti_format/' #需要修改成自己的root
# data_prefix = dict(
#     pts='samples/LIDAR_TOP',
#     CAM_FRONT='samples/CAM_FRONT',
#     CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
#     CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
#     CAM_BACK='samples/CAM_BACK',
#     CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
#     CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
#     sweeps='sweeps/LIDAR_TOP')
input_modality = dict(use_lidar=True, use_camera=False)
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/':
#         's3://openmmlab/datasets/detection3d/nuscenes/',
#         'data/nuscenes/':
#         's3://openmmlab/datasets/detection3d/nuscenes/',
#         './data/nuscenes_mini/':
#         's3://openmmlab/datasets/detection3d/nuscenes/',
#         'data/nuscenes_mini/':
#         's3://openmmlab/datasets/detection3d/nuscenes/'
#     }))
backend_args = None

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=150000,
            voxelize_reduce=True)), #点云体素化
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),# 体素化特征提取
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=grid_size,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1,1,0)), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=len(class_names),
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        train_cfg=dict(
            dataset='Waymo',
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.05,
            min_radius=1,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.6),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=2.0),
                iou_cost=dict(type='IoU3DCost', weight=2.0))),
        test_cfg=dict(
            dataset='Waymo',
            grid_size=grid_size,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type=None,
            score_thr=0.01,  # 确保保留低分框做合并
            max_per_img=1000  # 避免过早裁剪
        ),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=point_cloud_range,
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=8),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=2.0)))

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'waymo_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    classes=class_names,
    sample_groups=dict(Car=4, Pedestrian=12, Cyclist=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4],
        norm_intensity=True,
        norm_elongation=True,
        backend_args=backend_args))
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=(
            'sample_idx', 'timestamp', 'context_name', 'ego2global',
            'lidar_points', 'lidar_path', 'cam2img', 'ori_cam2img',
            'lidar2cam', 'lidar2img', 'cam2lidar', 'ori_lidar2img',
            'img_aug_matrix', 'box_type_3d', 'img_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix',
            'num_pts_feats'
        )
    )
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=6,
        use_dim=5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[1., 1.],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.0, flip_ratio_bev_vertical=0.0),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=(
            'sample_idx', 'timestamp', 'context_name', 'ego2global',
            'lidar_points', 'lidar_path', 'cam2img', 'ori_cam2img',
            'lidar2cam', 'lidar2img', 'cam2lidar', 'ori_lidar2img',
            'img_aug_matrix', 'box_type_3d', 'img_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix',
            'num_pts_feats'
        )
    )
]
# dataset_type = 'WaymoDataset'
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',               # ✅ 重复数据集
        times=1,                           # 建议 10～20；10 已经很够用
        dataset=dict(
            type='WaymoDataset',
            data_root=data_root,
            ann_file='waymo_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            load_interval=1,
            backend_args=backend_args
        )
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='training/velodyne', sweeps='training/velodyne'),
        ann_file='waymo_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='WaymoMetric', waymo_bin_file='/home/wangyubo/data/Waymo_mini/waymo_format/gt_mini.bin',
    result_prefix='./mini_pred',
    metric='mAP')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# For waymo dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
lr = 5e-5

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01, betas=(0.9, 0.99)),
    clip_grad=dict(max_norm=10.0, norm_type=2)   # 0.1 会太狠，设 10 更稳
)

# 原：lr_config/momentum_config 为 cyclic，target_ratio=(10, 1e-4) 与 (0.894736..., 1)，step_ratio_up=0.4，total_epochs=36
param_scheduler = [
    # LR up: 0 -> 10 * lr
    dict(
        type='CosineAnnealingLR',
        T_max=14.4,
        eta_min=lr * 10,
        begin=0,
        end=14.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # LR down: 10 * lr -> 1e-4 * lr
    dict(
        type='CosineAnnealingLR',
        T_max=21.6,
        eta_min=lr * 1e-4,
        begin=14.4,
        end=36,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # Momentum up: 0 -> 0.85/0.95
    dict(
        type='CosineAnnealingMomentum',
        T_max=14.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=14.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # Momentum down: 0.85/0.95 -> 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=21.6,
        eta_min=1,
        begin=14.4,
        end=36,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=36, val_interval=36)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=2)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=10))
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]
