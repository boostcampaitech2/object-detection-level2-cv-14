'''
Optimizer, Learning Schedule Settings
- optimizer 수정
- lr scheduler 수정 
- epoch 설정 
'''

# optimizer
optimizer = dict(type='Adam', lr=0.0002)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing', 
    by_epoch=False,
    warmup='linear', # 'linear','exp', 'constant'
    warmup_iters=500, 
    warmup_ratio=0.001, # The ratio of the starting learning rate used for warmup
    min_lr=1e-07)
runner = dict(type='EpochBasedRunner', max_epochs=32)