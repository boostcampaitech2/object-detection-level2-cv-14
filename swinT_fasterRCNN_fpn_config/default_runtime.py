'''
1. WandbLoggerHook 를 추가했습니다. https://github.com/boostcampaitech2/object-detection-level2-cv-14/discussions/6
'''
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',interval=1000, 
            init_kwargs=dict(
                project='fasterRCNN_r50_fpn',
                name = 'exp6_cosineAnnealing') #'실험할때마다 RUN에 찍히는 이름'
            )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook'),
               ]


dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
