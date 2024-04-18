_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/arm9.py'
]
evaluation = dict(interval=1, metric=['mpjpe', 'p-mpjpe'], save_best='MPJPE')

# optimizer settings
optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='exp',
    by_epoch=True,
    gamma=0.975,
)

total_epochs = 160

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=9,
    dataset_joints=9,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8
    ])

# model settings
model = dict(
    type='PoseLifter',
    pretrained=None,
    backbone=dict(
        type='TSGCN_arm9',
        in_channels=2,
        stem_channels=256,  # 1024
        num_blocks=3,
        kernel_sizes=(3, 3, 3, 3),
        dropout=0.25,
        num_joints=9,
        use_stride_conv=True),  # True
    keypoint_head=dict(
        type='TemporalRegressionHead',
        in_channels=256,  
        num_joints=9,
        loss_keypoint=dict(type='MPJPELoss')),
    train_cfg=dict(),
    test_cfg=dict(restore_global_position=True))   # 恢复全局位置

# data settings
data_root = '/media/star/zjurobot/402/WYH_402/ZJUHRI_Dataset/Arm9_3d/'
data_cfg = dict(
    num_joints=9,
    seq_len=81,
    seq_frame_interval=1,
    causal=False,
    temporal_padding=True,
    joint_2d_src='gt',
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotation_body3d/cameras.pkl',
)

train_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    dict(
        type='RelativeJointRandomFlip',
        item=['input_2d', 'target'],
        flip_cfg=[
            dict(center_mode='static', center_x=0.),
            dict(center_mode='root', center_index=0)
        ],
        visible_item=['input_2d_visible', 'target_visible'],
        flip_prob=0.5),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position'])
]

val_pipeline = [
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=[('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position'])
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=64,  # 128
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='Body3DArm9Dataset',
        ann_file=f'{data_root}/annotation_body3d/fps30/arm9_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='Body3DArm9Dataset',
        ann_file=f'{data_root}/annotation_body3d/fps30/arm9_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='Body3DArm9Dataset',
        ann_file=f'{data_root}/annotation_body3d/fps30/arm9_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
