''''''
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
classes = ( "General trash","Paper","Paper pack","Metal","Glass","Plastic", "Styrofoam","Plastic bag","Battery","Clothing")



img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augumentation 추가하지 않고 Pad는 제외한 상태
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True), #resize 조정
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True), #resize 조정
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            #dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8, #batch_size = samples_per_gpu*gpu개수 = 8
    workers_per_gpu=2, #데이터로더의 num workers와 동일
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_valid_split/new_train.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train_valid_split/new_valid.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
