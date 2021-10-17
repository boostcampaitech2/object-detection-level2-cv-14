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

<details>
<summary>Training</summary>


```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
  
  위 포맷을 따라 코드 실행 - 실제 사용 커맨드 :
  
```bash
$ python train.py --batch 8 
                  --epochs 50 
                  --data /opt/ml/detection/dataset/data.yaml 
                  --cfg ./models/yolov5s.yaml 
                  --weights yolov5s.pt 
                  --name yolov5x_50epochs_img640
```
  
  
  
</details>



<details>
<summary>Inference with detect.py</summary>


```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
  
  위 포맷을 따라 코드 실행 - 실제 사용 커맨드 :
  
```bash
$ python detect.py --weights /opt/ml/yolov5/yolov5/runs/train/yolov5x_50epochs/weights/best.pt 
                   --img 640
                   --conf 0.05 
                   --source /img_path
```
  

  
  
  
  
  
</details>

<details open>
<summary>Inference with inference_submission.py</summary>

  
detect.py를 한 후 submission을 위한 csv파일 생성 코드 






