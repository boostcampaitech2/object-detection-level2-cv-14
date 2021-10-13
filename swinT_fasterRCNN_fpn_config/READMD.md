# config 요약

`dataset.py`
- CLAHE augmentation 추가
- train, valid pipeline 을 fold2로 변경
- batch_size = 8


`faster_rcnn_r50_fpn.py`
- detector로 faster RCNN 이용
- neck으로 fpn 이용


`default_runtime.py`
- wandb hook 추가


`schedule_2x.py`
- lr scheduler : CosineAnnealing
- optimizer : Adam(lr=0.0002)
- warmup


`final_config.py`
- backbone을 swinT로 변경
- evaluation config : bbox_mAP_50 기준 best 모델 저장, classwise=True로 클래스별 valid AP 확인
- checkpoint_config : 모델 저장하는 에폭 interval, 모델 output dir 지정

학습 방법
- swinT_fasterRCNN_fpn_config 폴더는 detection/mmdetection/configs/ 아래 두시면 됩니다.
- 다음 코드로 학습하면 됩니다.
`python tools/train.py configs/swinT_fasterRCNN_fpn_config/final_config.py --seed=2021`
