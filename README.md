## 재활용 쓰레기 Object Detection

<img src="https://user-images.githubusercontent.com/44287798/137435947-efc7e013-6f00-4dee-b8fa-0fc4b255de89.png" width="300">  <img src="https://user-images.githubusercontent.com/44287798/137436777-3cacccf1-84a0-45df-86ba-42a207436bf2.png" width="310">

### 프로젝트 개요

우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있고 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있다.
 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나로, 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문이다.
 
우리는 사진에서 쓰레기를 탐지하는 모델을 만들어 이러한 문제점을 해결해보고자 한다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋을 사용한다. 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것이다.

### 팀원 소개
팀명: Machine==우리조 

||이름|역할|github|
|--|------|---|---|
|😙|김범수|EDA, Cascade R-CNN 수행|https://github.com/HYU-kbs|
|🤗|김준태|Faster R-CNN 실험|https://github.com/sronger|
|😎|김지성|공용 도구 개발, YOLO 실험, 앙상블 실험|https://github.com/intelli8786|
|😆|백종원|YOLO 실험, EfficientDet 실험|https://github.com/Baek-jongwon|
|😊|정소희|Faster R-CNN, Test time augmentation 실험|https://github.com/SoheeJeong|
|😄|홍지연|Faster R-CNN 성능 개선|https://github.com/hongjourney|


### 모델 성능 및 config file -> 각자 config file 올려주신 후 표에 적어주세요!! 

학습된 모델에 대한 설명과 성능, 각 모델에 대한 config file의 위치를 표로 나타내었다.
config file은 hyperparameter, model architecture, optimizer, scheduler, train/test dataset 등 모델에 대한 전반적인 학습 정보를 포함한다. 

학습, 추론을 위해 mmdetection library를 설치해야 한다. [링크](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)를 참조.

|모델|mAP50|config|
|------|---|---|
|SwinTransformer, FasterR-CNN|0.530|[config](https://github.com/boostcampaitech2/object-detection-level2-cv-14/tree/main/configs/swinT_fasterRCNN_fpn_config)|
|SwinTransformer, FasterR-CNN, MultiScaleTTA|0.531|[config](https://github.com/boostcampaitech2/object-detection-level2-cv-14/tree/main/configs/swinT_fasterRCNN_fpn_MultiScale_config)|
|SwinTransformer, FasterR-CNN, TileAugTTA|0.530|[config](https://github.com/boostcampaitech2/object-detection-level2-cv-14/tree/main/configs/swinT_fasterRCNN_fpn_TileAug_config)|
|YOLOv4|0.473|[config](https://github.com/boostcampaitech2/object-detection-level2-cv-14/tree/main/configs/YOLOv4_Darknet)|
|모델|xx.xxx|link to config|




### 실행파일 설명

**DataRefiner**: 데이터를 분리하고 가공하는 모듈이다. 
* 실행방법: ```python DataRefiner.py```
* DataRefiner.py 코드의 기능:
  - Class imbalance를 줄이기 위해 각 클래스별 데이터 양을 균등하게 조정해 train, validation 데이터를 분할한다.
  - Detector가 분류하기 어려운 크기(factor_minSize)를 제거한다.
  - Train, Validation 데이터를 일정 비율(factor_trainValRatio)로 분리한다.
  - COCO, Darknet 포맷으로 Annotation 데이터를 저장한다.

**DataVisualization**: 추론 결과를 눈으로 확인하기 위한 모듈이다.
* 실행방법: 
  - dependency 설치: ```pip install opencv-python```
  - 실행: ```python VisualizeEvaluation.py```
* VisualizeEvaluation.py 코드의 기능:
  - 모델이 Test 데이터를 추론한 결과를 눈으로 확인하기 위한 목적으로 제작되었다.
  - GUI 환경에서 작동되므로, 콘솔환경에서는 사용할 수 없다.
  - Visualization 타겟은 Target Image, Bounding Box, Confidence, Class Name 이다.
* 이 코드는 다음 기능으로 프로세스를 컨트롤 할 수 있다.
  - right Key : 인덱스를 1칸 앞으로 이동시킨다.
  - left Key : 인덱스를 1칸 뒤로 이동시킨다.
  - up Key : 인덱스를 10칸 앞으로 이동시킨다.
  - down Key : 인덱스를 10칸 뒤로 이동시킨다.
  - esc Key : 프로세스를 종료한다.
  - S Key : 현재 표시되는 시각화된 이미지를 저장한다.

**configs**: 각 모델에 대한 config 파일이 담겨있는 폴더이다.
