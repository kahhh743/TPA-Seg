# 使用 custom_imports 将 mmpretrain 的 models 添加进注册器

_base_ = [
    'mmpretrain::_base_/schedules/imagenet_bs4096_AdamW.py',
    'mmpretrain::_base_/datasets/imagenet_bs64_pil_resize.py',
    'mmpretrain::_base_/default_runtime.py',
    'mmpretrain::_base_/models/vit-base-p16.py',
]
custom_imports = dict(imports=['mmseg.models'], allow_failed_imports=False)
norm_cfg = dict(type='BN', requires_grad=True)
'''model'''
# model settings
model = dict(
    _scope_='mmseg',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        pre_norm=True,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='mmseg.UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ########
    init_cfg=dict(type="mmpretrain.Pretrained",checkpoint='')
)


'''dataset'''
# dataset settings
#dataset_type = 'ImageNet'

dataset_type = 'ConsepTypeDataset'
data_preprocessor = dict(
    num_classes=8,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        ##########
        data_root='data/consep',
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label'),
        ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep',
        data_prefix=dict(
            img_path='val/img', seg_map_path='val/label'),
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

# If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep',
        data_prefix=dict(
            img_path='test/img', seg_map_path='test/label'),
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator



'''optimizer'''
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.3,
                   clip_grad=dict(max_norm=1.0)),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
)
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=30,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=270,
        by_epoch=True,
        begin=30,
        end=300,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)



'''runtime'''
# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=5),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)



'''
# 使用 custom_imports 将 mmpretrain 的 models 添加进注册器
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

'''
#model
'''
# model settings
model = dict(
    _scope_='mmpretrain',
    type='Clip',
    backbone=dict(
        type='VisionTransformer',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        pre_norm=True,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
    ),
    ########
    init_cfg=dict(type="Pretrained",checkpoint='')
)


'''
#dataset
'''
# dataset settings
dataset_type = 'ImageNet'
#dataset_type = 'Ink-ConsepDataset'
data_preprocessor = dict(
    num_classes=8,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=224,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        ##########
        data_root='data/consep/train/img',
        # ann_file='meta/train.txt',
        # data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep/val/img',
        # ann_file='meta/val.txt',
        # data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep/test/img',
        # ann_file='meta/val.txt',
        # data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator



'''
#optimizer
'''
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.3,
                   clip_grad=dict(max_norm=1.0)),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
)
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=30,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=270,
        by_epoch=True,
        begin=30,
        end=300,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)

'''
#runtime
'''
# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=10),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

'''