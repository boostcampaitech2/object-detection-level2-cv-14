_base_ = [
    'faster_rcnn_r50_fpn.py',
    'dataset.py',
    'schedule_2x.py', 'default_runtime.py'
]

#swin transformer 덮어씌우기
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  
model = dict(
    #type='MaskRCNN', # detector로 우리가 지정한 faster_rcnn_r50_fpn을 사용할거라 이부분 지우고 
    backbone=dict(
        _delete_=True, # 백본이었던 resnet을 지우고 swin transformer를 사용하겠다. 
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


# valuation metric 설정
evaluation = dict(
        interval = 1, # valid로 evaluation하는 interval(epoch기준)
        metric = 'bbox', # evaluation동안 metric
        save_best = 'bbox_mAP_50', # bbox_mAP_50이 
        classwise = True # evalutation 마다 
)

#checkpoint config 설정
# 학습 초반에는 bbox_mAP_50이 한동안 증가하므로 에폭마다 저장됨. faster rcnn은 보통 30에폭 넘으면 수렴해서 interval을 3으로 줘도 괜찮을듯. 
checkpoint_config= dict(
        #interval=1, # interval=1하면 에폭마다 모델이 저장됨. 이 부분을 없애면 best모델이 계속해서 하나의 파일에 덮어쓰기로 저장됨.
        out_dir = '/opt/ml/detection/mmdetection/work_dirs/fold3_aug_swinT_fasterRCNN',# 모델 저장하는 위치(따로 지정 안 하면 현재 config파일 이름으로 새 work_dir 폴더가 만들어짐)
)
