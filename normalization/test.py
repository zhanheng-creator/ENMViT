_base_ = [
    '../../_base_/datasets/mmseg/medicalCells.py',
    '../../_base_/mmseg_runtime_my.py',
    '../../_base_/schedules/mmseg/schedule_epoch.py'
]

# norm_cfg = dict(type='SyncBN', requires_grad=True)
norm_cfg = dict(type='BN', requires_grad=True)

# pspnet r18
student = dict(
    type='mmseg.EncoderDecoder',
    backbone=dict(
        type='hrvit_b0',
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
    ),
    # 主解码头
    decode_head=dict(
        type='UPerHead',
        # 输入通道数
        in_channels=[16, 32, 64, 128],
        # 被选择的特征图(feature map)的索引
        in_index=[0, 1, 2, 3],
        # 平均池化的规模，需要看文章
        pool_scales=(1, 2, 3, 6),
        # 解码头中间态(intermediate)的通道数
        channels=512,
        # 进入最后分类层(classification layer)之前的 dropout 比例
        dropout_ratio=0.1,
        # 预测类别数
        num_classes=2,
        norm_cfg=norm_cfg,
        # 这里的align_corners，还有下面的就是上采样的时候，是将x的图片，变成2x，还是2x+1
        # 按照我看的资料，结果为True可以更好的解决边角问题，可以尝试修改，注意上下采样要用都要用
        align_corners=False,
        # 损失函数
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # 辅助解码头
    auxiliary_head=dict(
        type='FCNHead',
        # 特征图的输入维度
        in_channels=64,
        # 以倒数第二层，这样一个低层次的特征作为输入
        in_index=2,
        # 卷积层的通道数
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        # 低层次特征的loss计算，可以鼓励主干网络训练出更好的低层次特征
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    # model training and testing settings
    # 定义模型训练的行为
    train_cfg=dict(),
    # 定义模型测试的行为
    # 这个是把整张图片都丢到模型里训练，还有一个slide模式，是图片太大，你把他切成一小块一小块，丢进去
    test_cfg=dict(mode='whole')
)

checkpoint = '/root/autodl-tmp/mmsegmentation--500/tools/checkpoint20230104_222709/output/epoch_500.pth'  # noqa: E501

# pspnet r101
teacher = dict(
    type='mmseg.EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    backbone=dict(
        type='hrvit_b0',
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
        with_cp=False
    ),
    # 主解码头
    decode_head=dict(
        type='UPerHead',
        # 输入通道数
        in_channels=[16, 32, 64, 128],
        # 被选择的特征图(feature map)的索引
        in_index=[0, 1, 2, 3],
        # 平均池化的规模，需要看文章
        pool_scales=(1, 2, 3, 6),
        # 解码头中间态(intermediate)的通道数
        channels=512,
        # 进入最后分类层(classification layer)之前的 dropout 比例
        dropout_ratio=0.1,
        # 预测类别数
        num_classes=2,
        norm_cfg=norm_cfg,
        # 这里的align_corners，还有下面的就是上采样的时候，是将x的图片，变成2x，还是2x+1
        # 按照我看的资料，结果为True可以更好的解决边角问题，可以尝试修改，注意上下采样要用都要用
        align_corners=False,
        # 损失函数
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # 辅助解码头
    auxiliary_head=dict(
        type='FCNHead',
        # 特征图的输入维度
        in_channels=64,
        # 以倒数第二层，这样一个低层次的特征作为输入
        in_index=2,
        # 卷积层的通道数
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        # 低层次特征的loss计算，可以鼓励主干网络训练出更好的低层次特征
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4))
)

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMSegArchitecture',
        model=student,
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module=('decode_head.conv_seg','auxiliary_head.conv_seg'),
                teacher_module=('decode_head.conv_seg','auxiliary_head.conv_seg'),
                # student_module='decode_head.conv_seg',
                # teacher_module='decode_head.conv_seg',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_logits',
                        tau=3,
                        loss_weight=5,
                    )
                ])
        ]),
)

find_unused_parameters = True