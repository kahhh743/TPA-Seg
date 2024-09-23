_base_ = [
    'mmpretrain::_base_/schedules/imagenet_bs4096_AdamW.py',
    'mmpretrain::_base_/datasets/imagenet_bs64_pil_resize.py',
    'mmpretrain::_base_/default_runtime.py',
    'mmpretrain::_base_/models/vit-base-p16.py',
]

model = dict(
    head=dict(
        num_classes=8,
    ),
    ########
    init_cfg=dict(type="Pretrained", checkpoint='ckpt/clip-vit-base-p16_openai-pre_3rdparty_in1k_20221220-c7d9c899.pth ')
)

data_preprocessor = dict(
    num_classes=8,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        ##########
        data_root='data',
        # ann_file='meta/train.txt',
        # data_prefix='train',
        ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(
        data_root='data',
        # ann_file='meta/val.txt',
        # data_prefix='val',
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=5,
    dataset=dict(

        data_root='data',
        # ann_file='meta/val.txt',
        # data_prefix='val',
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)