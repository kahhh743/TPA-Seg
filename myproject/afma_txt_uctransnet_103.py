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
            # dict(type='CrossEntropyLoss', loss_name='loss_ce',
            #      loss_weight=1.0, use_sigmoid=False, class_weight=[0.5, 2.0, 2.0, 2.0]),
            # dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=1.0)
            # dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=2.0, class_weight=[0.5, 2.0, 2.0, 2.0])
            # dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=1.0, class_weight=[0.5, 2.0, 2.0, 2.0],
            #      alpha=0.5, beta=0.5,)
            # dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, use_sigmoid=False,
            #      class_weight=[1/(7*0.8397), 1/(7*0.0013), 1/(7*0.1031), 1/(7*0.0185), 1/(7*0.0055), 1/(7*0.001), 1/(7*0.0308)]),
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, use_sigmoid=False,
                 class_weight=[1 / (5 * 0.8361), 1 / (5 * 0.0017), 1 / (5 * 0.0266), 1 / (5 * 0.0864),
                               1 / (5 * 0.0491)]),
            dict(type='TverskyLoss', loss_name='loss_tversky', alpha=0.5, beta=0.5)
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
img_scale = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='Resize', scale=img_scale, keep_ratio=False),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='MyRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    # dict(type='PackSegInputs'),
    dict(type='LoadImgText'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='PackSegInputs'),
    dict(type='LoadImgText'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    # persistent_workers=True,
    # sampler=dict(type='InfiniteSampler', shuffle=True),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # data_prefix=dict(
        #     img_path='train/img', seg_map_path='train/label-c5', text_path='train/label-c5/train_text_c5_class+local+size.txt'),
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label-c5', text_path='train/label/train_text_c5.txt'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/img', seg_map_path='val/label-c5', text_path='val/label/val_text_c5.txt'),
        # data_prefix=dict(
        #     img_path='val/img', seg_map_path='val/label-c5', text_path='val/label-c5/val_text_c5_class+local+size.txt'),
        pipeline=test_pipeline))


val_evaluator = [
    dict(type='MyIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
    # dict(type='SegClsAcc')
]
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    # persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/img', seg_map_path='test/label-c5', text_path='test/label-c5/test_text_c5.txt'),
        pipeline=test_pipeline))
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
                    # paramwise_cfg=dict(
                    #     custom_keys={
                    #         'absolute_pos_embed': dict(decay_mult=0.),
                    #         'relative_position_bias_table': dict(decay_mult=0.),
                    #         'norm': dict(decay_mult=0.),
                    #         '.cls_token': dict(decay_mult=0.0),
                    #         '.pos_embed': dict(decay_mult=0.0)
                    #     })
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

# training schedule for 160k
# train_cfg = dict(
#     type='IterBasedTrainLoop', max_iters=10000, val_interval=100)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)



'''runtime'''
default_scope = 'mmseg'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', save_best='mDice_nobg'),
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
