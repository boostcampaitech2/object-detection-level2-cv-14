## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Inference</summary>

Inference with YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36). Models automatically download
from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>



<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading models automatically from
the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

Run commands below to reproduce results
on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on
first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the
largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).

```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">



