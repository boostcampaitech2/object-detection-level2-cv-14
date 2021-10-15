* 학습 방법
  - 각 config 폴더는 detection/mmdetection/configs/ 아래 둔다.
  - 다음과 같이 학습한다. ```python tools/train.py configs/swinT_fasterRCNN_fpn_MultiScale_config/final.py```

* 추론 방법
  - ```python tools/test.py configs/swinT_fasterRCNN_fpn_MultiScale_config/final.py path_to_your_weightfile/weight.pth --work_dir=work_dirs/tile_cropsize --eval=bbox```