dataset_type = 'CocoDataset'
data_root = '../dataset/'
annotation_root = '../Annotations/coco/'
classes = ["General trash","Paper","Paper pack","Metal","Glass","Plastic","Styrofoam","Plastic bag","Battery","Clothing"]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.01,
        scale_limit=0.0,
        rotate_limit=0.03,
        interpolation=1,
        p=0.5),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TileAug',
        img_scale=(1024, 1024),
        crop_size=(512,512),
        flip=False,
        transforms=[
            dict(type='TileImage'),#, img_scale=(1024,1024)),
            dict(type='Resize', keep_ratio=True),#, img_scale=(1024,1024)),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            #dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

#testing example: python tools/test.py configs/config_custom/final.py work_dirs/final/epoch_32.pth --work_dir=work_dirs/final --eval=bbox

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=annotation_root + 'fold_1_train.json',
        img_prefix=data_root +'train',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=annotation_root + 'fold_1_valid.json',
        img_prefix=data_root+'train',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=annotation_root + 'test.json'
        img_prefix=data_root
        classes=classes,
        pipeline=test_pipeline))

evaluation = dict(
    interval=2, 
    metric='bbox',
    save_best ='bbox_mAP_50',
    classwise=True)
