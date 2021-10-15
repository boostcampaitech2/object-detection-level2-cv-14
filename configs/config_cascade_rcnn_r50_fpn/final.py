#config들을 합쳐줌
_base_ = [
   'cascade_rcnn_r50_fpn.py',
   'albu_dataset.py',
   'default_runtime.py',
   'schedule_1x.py'
]
#python tools/train.py configs/config_custom/final.py