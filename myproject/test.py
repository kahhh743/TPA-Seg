_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (224, 224)
data_preprocessor = dict(size=crop_size)
model = dict(
    backbone=dict(img_size=(224, 224)),
    data_preprocessor=data_preprocessor,
    pretrained='',
    decode_head=dict(num_classes=8),
    auxiliary_head=dict(num_classes=8))


# dataset settings
dataset_type = 'ConsepTypeDataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='LoadAnnotations'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
    batch_size=8,
    num_workers=5,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep',
        pipeline=train_pipeline,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label'),
        ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=5,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep',
        pipeline=test_pipeline,
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
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep',
        pipeline=test_pipeline,
        data_prefix=dict(
            img_path='test/img', seg_map_path='test/label'),
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]



'''
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=(224, 224),
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    #pretrained='pretrain/jx_vit_base_p16_224-80ecf9dd.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(224, 224),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic'),
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[768, 768, 768, 768],
        out_channels=768,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
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
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=8,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable






# data settings

# dataset settings
dataset_type = 'ConsepTypeDataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
    batch_size=8,
    num_workers=5,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep',
        pipeline=train_pipeline,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label'),
        ),
    sampler=dict(type='DefaultSampler', shuffle=True),

)

val_dataloader = dict(
    batch_size=8,
    num_workers=5,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep',
        pipeline=train_pipeline,
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
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep',
        pipeline=train_pipeline,
        data_prefix=dict(
            img_path='test/img', seg_map_path='test/label'),
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator


# schedule setting

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict()
test_cfg = dict()
# optimizer

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.3,
                   ),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
# training schedule for 160k

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))



# default runtime
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
'''