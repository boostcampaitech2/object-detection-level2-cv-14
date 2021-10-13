* __init__.py, test_time_aug.py, transforms.py 는 mmdet/pipelines 아래에 있는 파일로, 기존 파일에 tile augmentation을 위한 코드를 추가한 것입니다.
* tile_aug_dataset_config_example.py 는 tile augmentation을 사용할 때 dataset config 의 예시입니다.
* 현재 코드는 1024*1024 -> 512*512 4개로 쪼개는 것으로 사이즈가 고정되어 있습니다.
