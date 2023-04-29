# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# dataset settings 数据集配置
dataset_type = 'ADE20KDataset'
data_root = '../data/ade/ADEChallengeData2016'
#data_root = '../data/myData'
img_norm_cfg = dict(   #图像归一化配置，用来归一化输入的图像
    mean=[123.675, 116.28, 103.53], # 预训练里用于预训练主干网络模型的平均值。
    std=[58.395, 57.12, 57.375], # 预训练里用于预训练主干网络模型的标准差。
    to_rgb=True) # 预训练里用于预训练主干网络的图像的通道顺序。
crop_size = (512, 512)# 训练时的裁剪大小
#训练过程数据处理流水化。数据集预处理 Pipeline
train_pipeline = [
    #数据读取，把图像和对应的标注图读进来
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    #数据增强
    dict(type='Resize', #调整图片大小，同时需要将分割图的大小调整变化图像和其注释大小的数据增广。
         img_scale=(2048, 512),  # 图像和标注的 resize 尺度
         ratio_range=(0.5, 2.0)), # 随机 resize 的比例范围。
    dict(type='RandomRotate', prob=1, degree=(-180, 180)),  # 随机旋转
    dict(type='RandomCrop',#对图片进行随机裁减，大概率裁剪掉噪音，防止过拟合，随机裁剪当前图像和其注释。
         crop_size=crop_size,# 随机裁剪图像生成 patch 的大小。
         cat_max_ratio=0.75),# 单个类别可以填充的最大区域的比例。
    dict(type='RandomFlip', prob=0.5),                                    #对图像进行随机翻转，翻转图像的概率
    dict(type='PhotoMetricDistortion'),                                   #可以调整图像的亮度，色度，对比度，饱和度，以及加入噪点，光度失真  光学上使用一些方法扭曲当前图像。
    dict(type='Normalize', **img_norm_cfg),                               #归一化
    #格式化处理
    dict(type='Pad',  # 填充当前图像到指定大小。
         size=crop_size, # 填充的图像大小。
         pad_val=0,  # 图像的填充值。
         seg_pad_val=255), # 'gt_semantic_seg'的填充值。
    dict(type='DefaultFormatBundle'), # 默认格式转换的组合操作。
    dict(type='Collect', keys=['img', 'gt_semantic_seg']), # 决定数据里哪些键被传递到分割器里的流程。
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
#真正的数据读取过程，上面的实际上也是参数而已
data = dict(
    samples_per_gpu=4, #相当于batch size
    workers_per_gpu=4, #数据加载时每个 GPU 使用的子进程（subprocess）数目。0 则意味着主进程加载数据。    注意，这两个参数仅仅在训练时有效，
    #数据处理流水线，设置路径之类
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))
