'''
Final Config
- Base model의 backbone, neck 수정
- evaluation 설정 수정 
- checkpoint 설정 수정 
'''

# Base Config
_base_ = [
    'faster_rcnn_r50_fpn.py',
    'dataset.py',
    'schedule_2x.py', 'default_runtime.py'
]

# backbone 수정 
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  
model = dict(
    # type='MaskRCNN', # detector 제외 (Base model의 faster RCNN사용하기 위함)
    backbone=dict(
        _delete_=True, # backbone인 ResNet 제외,  swin transformer로 수정 
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))


# evaluation 설정
evaluation = dict(
        interval = 1, # valid set evaluation interval
        metric = 'bbox', # evaluation metric
        save_best = 'bbox_mAP_50', # best model 저장 지표 bbox_mAP_50
        classwise = True # class별 AP확인 
)


#checkpoint config 설정
checkpoint_config= dict(
        # interval=1, # interval을 1로 설정하면 best모델 상관 없이 1 epoch마다 저장됨
        out_dir = '/opt/ml/detection/mmdetection/work_dirs/fold3_aug_swinT_fasterRCNN',# 모델 저장 위치(따로 지정 안 하면 현재 config 파일 이름으로 새 work_dir 폴더가 생성됨)
)
