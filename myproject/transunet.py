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
resnet = dict(
    num_layers=(3, 4, 9),
    width_factor = 1,
)
transformer = dict(
    mlp_dim = 3072,
    num_heads = 12,
    num_layers = 12,
    attention_dropout_rate = 0.0,
    dropout_rate = 0.1,
)
model = dict(
    type='HoverEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='VisionTransformerUnet',
        img_size=224,
        config=dict(resnet=resnet,
                    hidden_size = 768,
                    transformer=transformer,
                    classifier = 'seg',
                    pretrained_path = 'D:/PyCharmProject/HZ/own-TransUNet/config_vit.pretrained_path/R50+ViT-B_16' \
                             '.npz ',
                    decoder_channels = (256, 128, 64, 16),
                    skip_channels = [512, 256, 64, 16],
                    n_classes = 4,
                    n_skip = 3,
                    activation = 'softmax',
                    patches=dict(size = (16, 16),
                                 grid = (14, 14)
                                 ),
                    representation_size = None,
                    resnet_pretrained_path = None,
                    patch_size = 16,
                    ),
        num_classes=4,
        ),
    decode_head=dict(
        type='My_FCNHead',
        in_channels=4,
        channels=4,
        num_classes=4,
        loss_decode=[
            # dict(type='CrossEntropyLoss', loss_name='loss_ce',
            #      loss_weight=1.0, use_sigmoid=False, class_weight=[0.5, 2.0, 2.0, 2.0]),
            # dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=1.0)
            dict(type='TverskyLoss', loss_name='loss_tversky', loss_weight=2.0, class_weight=[0.5, 2.0, 2.0, 2.0])
            # dict(type='TverskyLoss', loss_name='loss_tversky', alpha=0.5, beta=0.5)
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
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    # dict(type='MyRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
    # dict(type='LoadImgText'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),

    dict(type='PackSegInputs'),
    # dict(type='LoadImgText'),
]

train_dataloader = dict(
    batch_size=24,
    num_workers=12,
    # persistent_workers=True,
    # sampler=dict(type='InfiniteSampler', shuffle=True),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label-c4', text_path='train/label-c4/train_text_c4.txt'),
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
            img_path='val/img', seg_map_path='val/label-c4', text_path='val/label-c4/val_text_c4.txt'),
        pipeline=test_pipeline))


val_evaluator = [
    dict(type='MyIoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
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
    logger=dict(type='LoggerHook', interval=10),
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
