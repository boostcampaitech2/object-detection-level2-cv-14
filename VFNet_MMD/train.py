from multiprocessing import Process, freeze_support
from mmcv import Config
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)

def main():
    cfg = Config.fromfile('vfnet_x101_64x4d_fpn_mstrain_2x_coco.py')

    cfg.seed = 7
    cfg.gpu_ids = [0]
    cfg.work_dir = 'work_dirs/myModel'

    # build_dataset
    datasets = build_dataset(cfg.data.train)

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets, cfg, distributed=False, validate=False)

if __name__ == '__main__':
    freeze_support()
    main()