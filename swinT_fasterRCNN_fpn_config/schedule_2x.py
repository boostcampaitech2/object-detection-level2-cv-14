# optimizer
optimizer = dict(type='Adam', lr=0.0002)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing', 
    by_epoch=False,
    warmup='linear', # `exp`, `constant`도 가능
    warmup_iters=500, # warmup할 iteration
    warmup_ratio=0.001, # The ratio of the starting learning rate used for warmup
    min_lr=1e-07)
runner = dict(type='EpochBasedRunner', max_epochs=32) # max_epochs으로 32