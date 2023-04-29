# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# This source code is licensed(Dual License(GPL3.0 & Commercial)) under the license found in the
# LICENSE file in the root directory of this source tree.  Sync
# --------------------------------------------------------------------------------

_base_ = [
    '../_base_/models/upernet_mpvit.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
#modek字段定义一个模型，实际就是python的字典
model = dict(
    pretrained='https://dl.dropbox.com/s/1o07eti5rgve1i6/mpvit_small_mm.pth',
    # pretrained='null',
    backbone=dict(
        type='MPViT',
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[64, 128, 216, 288],
        drop_path_rate=0.2,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False
    ),
    #解码头
    decode_head=dict(
        in_channels=[128, 216, 288, 288],#输入维度
        num_classes=150, #分类的类别数
    ),
    #辅助解码头，学习低层次特征
    auxiliary_head=dict(
        in_channels=288,#输入维度
        num_classes=150
    ),
    train_cfg = dict(
        type = "mixup"
    )
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
