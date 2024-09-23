norm_cfg = dict(type='SyncBN', requires_grad=True)

# model settings
model = dict(
    type='CLIP',
    embed_dim=768,
    image_resolution=224,
    vision_layers=12,
    vision_width=768,
    vision_patch_size=16,
    context_length=77,
    vocab_size=49408,
    # transformer_width=768,
    # transformer_heads=12,
    transformer_layers=12,

    # embed_dim=512,
    # vision_patch_size=32,
    transformer_width=512,
    transformer_heads=8,

    finetune=False,
    average_targets=1,
    data_preprocessor=dict(
        type='MutiSegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        size=(224,224),
        pad_val=0,
        seg_pad_val=255),

    # data_preprocessor=dict(
    #     type='MultiModalDataPreprocessor',
    #     mean=[123.675, 116.28, 103.53],     # The pixel mean of R, G, B channels.
    #     std=[58.395, 57.12, 57.375],        # The pixel standard deviation of R, G, B channels.
    #     to_rgb=True,                    # whether to convert image from BGR to RGB.
    #     pad_value=0,                         # Padding value.
    #     pad_size_divisor=1
    # ),


)  # yapf: disable


# data settings

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadImgText'),

]

dataset_type = 'ConsepTypeDataset'
train_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        ##########
        data_root='data/consep',
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label', text_path='train/label/test_text.txt'),
        pipeline=train_pipeline,
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
            img_path='train/img', seg_map_path='train/label', text_path='train/label/test_text.txt'),
        pipeline=train_pipeline,
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
#val_evaluator = dict(type='Accuracy', top_k=(1, 5))


# If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/consep',
        data_prefix=dict(
            img_path='train/img', seg_map_path='train/label', text_path='train/label/test_text.txt'),
        pipeline=train_pipeline,
        ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator



'''optimizer'''
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.3),
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
train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=2)
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
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1)
)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(
    by_epoch=True,
)


log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
