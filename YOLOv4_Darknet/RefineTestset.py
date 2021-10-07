import json
import os

sourceAnnotation = 'C:/Dataset/Detection/dataset/test.json'  # 원시 train json 파일 경로를 지정해주세요
sourceImageDir = 'C:/Dataset/Detection/dataset/test'  # train 이미지폴더를 지정해주세요

with open(sourceAnnotation, 'r') as f:
    json_origin = json.load(f)

with open("darknet_test.txt", 'w') as f:
    for image_origin in json_origin["images"]:
        f.write(sourceImageDir +"/"+image_origin['file_name'].split('/')[1]+'\n')

