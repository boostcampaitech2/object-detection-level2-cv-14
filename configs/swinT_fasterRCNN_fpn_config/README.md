# Swin Transformer + Faster RCNN Config


## Dataset Settings
- `dataset.py`
- 주어진 문제에 맞는 classes 선언
- augmentation 수정 
- batch_size = samples_per_gpu x gpu개수
- train, valid, test pipeline 수정 

## Base Model Settings
- `faster_rcnn_r50_fpn.py`
- 주어진 문제에 맞게 num_classes = 10으로 변경
- detector로 Faster RCNN, neck으로 fpn 이용

## Runtime Settings
- `default_runtime.py`
- WandbLoggerHook 추가

## Optimizer, Learning Schedule Settings
- `schedule_2x.py`
- optimizer 수정
- lr scheduler 수정 
- epoch 설정 
- Cosine Annealing, Adam, warmup 이용

## Final Config
- `final_config.py`
- Base Model의 backbone, neck 수정
- evaluation 설정 : bbox_mAP_50 기준 best 모델 저장, 클래스 별 validation AP 확인
- checkpoint 설정 : 모델 저장 주기, 경로 지정 


## 학습 방법
- `python tools/train.py {final_config경로} --seed=2021`
- ex) `python tools/train.py configs/swinT_fasterRCNN_fpn_config/final_config.py --seed=2021`
