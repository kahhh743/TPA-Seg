# _base_ = [
#     '../_base_/models/upernet_swin.py', '../_base_/datasets/ade20k.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
# ]
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=1)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

data_preprocessor = dict(
    type='MutiSegDataPreProcessor',
    # type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size=(224,224),
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='SwinUnetEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SwinUnet',
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2, 2, 6, 2],
        num_classes = 3,
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        ),
    decode_head=dict(
        type='My_FCNHead',
        # in_channels=[96, 192, 384, 768],
        # in_index=[0, 1, 2, 3],
        # pool_scales=(1, 2, 3, 6),
        # channels=512,
        # dropout_ratio=0.1,
        num_classes=3,
        # norm_cfg=norm_cfg,
        # align_corners=False,
        in_channels=3,
        channels=3,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce',
                 loss_weight=1.0, use_sigmoid=False, class_weight=[0.1, 1.0, 1.0]),
            # dict(type='My_DiceLoss', loss_name='loss_dice', loss_weight=1.0),
            dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=2.0, class_weight=[0.1, 1.0, 1.0])
        ],
    ),
    auxiliary_head=dict(
        type='ClipforSwin',
        embed_dim=768,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
    ),

    # model training and testing settings

    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)


# dataset settings
dataset_type = 'ConsepTypeDataset'
data_root='data/consep'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),

    # dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PhotoMetricDistortion'),
    # dict(type='PackSegInputs'),
    dict(type='LoadImgText'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),

    # dict(type='PackSegInputs'),
    dict(type='LoadImgText'),
]

train_dataloader = dict(
    batch_size=12,
    num_workers=12,
    # persistent_workers=True,
    # sampler=dict(type='InfiniteSampler', shuffle=True),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label', text_path='train/label/test_text.txt'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=24,
    num_workers=8,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/img', seg_map_path='val/label', text_path='val/label/val_text.txt'),
        pipeline=test_pipeline))


val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
    # dict(type='SegClsAcc')
]
test_dataloader = val_dataloader
test_evaluator = val_evaluator

'''optimizer'''
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None,
                    paramwise_cfg=dict(
                        custom_keys={
                            'absolute_pos_embed': dict(decay_mult=0.),
                            'relative_position_bias_table': dict(decay_mult=0.),
                            'norm': dict(decay_mult=0.),
                            '.cls_token': dict(decay_mult=0.0),
                            '.pos_embed': dict(decay_mult=0.0)
                        })
)


# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.3),
#     # optimizer=optimizer,
#     # specific to vit pretrain
#     paramwise_cfg=dict(custom_keys={
#         '.cls_token': dict(decay_mult=0.0),
#         '.pos_embed': dict(decay_mult=0.0)
#     }),
# )

# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,
#         end=150000,
#         by_epoch=False)
# ]
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=1000,
        by_epoch=True)
]
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=15000,
    warmup_ratio=2e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# learning policy
# param_scheduler = [
#     # warm up learning rate scheduler
#     dict(
#         type='LinearLR',
#         start_factor=2e-6,
#         by_epoch=True,
#         begin=0,
#         end=100,
#         # update by iter
#         convert_to_iter_based=True),
#     # main learning rate scheduler
#     dict(
#         type='CosineAnnealingLR',
#         T_max=900,
#         by_epoch=True,
#         begin=100,
#         end=1000,
#     )
# ]

# training schedule for 160k
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)



'''runtime'''
default_scope = 'mmseg'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1),
    # adding
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),
                # dict(type='TensorboardVisBackend')
                ]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')



log_processor = dict(
    by_epoch=True,
)


log_level = 'INFO'
# load_from = pretrained
resume = False

tta_model = dict(type='SegTTAModel')
