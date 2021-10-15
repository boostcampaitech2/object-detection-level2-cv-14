_base_ = [
    'swint.py',
    'dataset.py',
    'default_runtime.py',
    'schedule_1x.py'
]

# swin transformer 덮어씌우기
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
