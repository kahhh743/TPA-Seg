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
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=(224,224),
    seg_pad_val=255)
model = dict(
    type='SwinUnetEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            # dict(type='CrossEntropyLoss', loss_name='loss_ce',
            #      loss_weight=1.0, use_sigmoid=False, class_weight=[0.5, 2.0, 2.0, 2.0]),
            dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=2.0, class_weight=[0.5, 5.0, 2.0, 2.0, 2.0])
            # dict(type='CrossEntropyLoss', loss_name='loss_ce',
            #      loss_weight=1.0, use_sigmoid=False),
            # dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=2.0)
        ],

    ),
    # auxiliary_head=dict(
    #     type='ClipforSwin',
    #     embed_dim=768,
    #     context_length=77,
    #     vocab_size=49408,
    #     transformer_width=512,
    #     transformer_heads=8,
    #     transformer_layers=12,
    # ),

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
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='MyRandomFlip', prob=0.5, direction='horizontal'),
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
    batch_size=48,
    num_workers=12,
    # persistent_workers=True,
    # sampler=dict(type='InfiniteSampler', shuffle=True),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label-c5', text_path='train/label-c5/train_text_c5.txt'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=12,
    num_workers=12,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/img', seg_map_path='val/label-c5', text_path='val/label-c5/val_text_c5.txt'),
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
    lr=0.0001,
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

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-4,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# learning policy
# param_scheduler = [
#     # warm up learning rate scheduler
#     dict(
#         type='LinearLR',
#         start_factor=1e-4,
#         by_epoch=True,
#         begin=0,
#         end=30,
#         # update by iter
#         convert_to_iter_based=True),
#     # main learning rate scheduler
#     dict(
#         type='CosineAnnealingLR',
#         T_max=270,
#         by_epoch=True,
#         begin=30,
#         end=300,
#     )
# ]
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False)
]
# training schedule for 160k
# train_cfg = dict(
#     type='IterBasedTrainLoop', max_iters=10000, val_interval=100)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
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
