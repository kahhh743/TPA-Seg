data = dict(samples_per_gpu=1)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='MutiSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=(224,224),
    seg_pad_val=255)
transformer = dict(
    num_heads=4,
    num_layers=4,
    embeddings_dropout_rate=0.1,
    attention_dropout_rate=0.1,
    dropout_rate=0
)
model = dict(
    type='AUTEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='AUTTransNet',
        img_size=224,
        config=dict(transformer=transformer,
                    expand_ratio=4,
                    patch_sizes=[16,8,4,2],
                    base_channel=64,
                    n_classes=1,
                    KV_size=1472,
                    with_text=True,
                    ),
        n_classes=5,
        ),
    decode_head=dict(
        type='AFMAHead',
        in_channels=5,
        channels=5,
        num_classes=5,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, use_sigmoid=False,
                 class_weight=[1 / (5 * 0.8195), 1 / (5 * 0.1207), 1 / (5 * 0.0415), 1 / (5 * 0.0025),
                               1 / (5 * 0.0159)]),
            dict(type='TverskyLoss', loss_name='loss_tversky', alpha=0.5, beta=0.5)
        ],

    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)


# dataset settings
dataset_type = 'Text_MoNuSACDataset'
data_root = 'data/MoNuSAC'
img_scale = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='MyRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='LoadImgText'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='LoadImgText'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=12,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label', text_path='train/label/train_text_c5.txt'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=24,
    num_workers=12,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/img', seg_map_path='val/label', text_path='val/label/val_text_c5.txt'),
        pipeline=test_pipeline))


val_evaluator = [
    dict(type='MyIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
]
test_dataloader = dict(
    batch_size=24,
    num_workers=12,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/img', seg_map_path='test/label', text_path='test/label/test_text_c5.txt'),
        pipeline=test_pipeline))
test_evaluator = val_evaluator

'''optimizer'''
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,

)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None,
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=100,
        # update by iter
        ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=290,
        by_epoch=True,
        begin=10,
        end=300,
    )
]

train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

'''runtime'''
default_scope = 'mmseg'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1),
    # adding
    visualization=dict(type='SegVisualizationHook', draw=False, interval=1)
)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend'),
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
