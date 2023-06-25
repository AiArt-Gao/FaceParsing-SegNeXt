norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoder',

    backbone=dict(
        type='MSCAN',
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.2,
        depths=[3, 3, 12, 3],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mscan_b.pth'),
    ),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=512,
        ham_channels=512,
        ham_kwargs=dict(MD_R=16),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

dataset_type = 'TFSDataset'
data_root = '/data2/jiangnan/dataset/TFS/TFS_GB'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='TFSDataset',
        data_root='/data2/jiangnan/dataset/TFS/TFS_GB',
        img_dir='train/images',
        ann_dir='train/labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            # dict(type='JNTrans'),
            # dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
            # dict(type='RandomCrop', crop_size=(473, 473), cat_max_ratio=0.75),
            dict(type='FaceTrans',
                 crop_size=[512, 512],
                 ),
            dict(type='RandomFlip', prob=0.),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',

                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            # dict(type='Pad', size=(473, 473), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='TFSDataset',
        data_root='/data2/jiangnan/dataset/TFS/TFS_GB',
        img_dir='val/images',
        ann_dir='val/labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='TFSDataset',
        data_root='/data2/jiangnan/dataset/TFS/TFS_GB',
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

# lr 0.00006
optimizer = dict(type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))
optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=60000)
evaluation = dict(interval=4000, metric=['mIoU', 'mFscore'], pre_eval=True)
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
work_dir = './work_dirs/segnext_TFS_base_GB'
gpu_ids = [1]
auto_resume = False