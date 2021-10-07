from multiprocessing import Process, freeze_support

import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test, init_detector
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np

epoch = 'epoch_24'

def main():
    cfg = Config.fromfile('vfnet_x101_64x4d_fpn_mstrain_2x_coco.py')

    cfg.seed = 7
    cfg.gpu_ids = [0]
    cfg.work_dir = 'work_dirs/myModel'
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    # build_dataset
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')


    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))  # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')  # ckpt load


    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05)  # output 계산

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)
    submission.head()

if __name__ == '__main__':
    freeze_support()
    main()