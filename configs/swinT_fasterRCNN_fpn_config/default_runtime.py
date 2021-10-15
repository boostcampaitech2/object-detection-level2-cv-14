'''
Runtime Settings
- WandbLoggerHook 추가
'''
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',interval=1000, 
            init_kwargs=dict(
                project='faster_RCNN', #project_name
                name = 'exp5_cosineAnealing') # exp_name
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
