import json
import os
import pandas as pd
import glob


txt_dir = "/opt/ml/yolov5/yolov5/runs/detect/exp12/labels/"
txt_list = glob.glob(txt_dir+'*.txt')
txt_list.sort()

prediction_strings = []
file_names = []
for txt in txt_list:
    # print(txt)
    txt_number = txt.split('/')[-1]
    # print(txt_number)
    image_id = 'test/'+txt_number.split('.')[-2] +'.jpg'
    print(image_id)
    prediction_string = ''
    with open(txt,'r') as f:
        lines = f.readlines()

        for line in lines:
            # print(line)
            info = line.strip().split(' ')

            cls = info[0]
            cx = float(info[1])*1024
            cy = float(info[2])*1024
            width = float(info[3])*1024
            height = float(info[4])*1024
            x1 = cx-width/2.0
            y1 = cy-height/2.0
            x2 = cx+width/2.0
            y2 = cy+height/2.0
            confidence = info[5]
            prediction_string += str(cls).strip() + ' ' + str(confidence).strip() + ' ' + str(x1).strip() + ' ' + str(y1).strip() + ' ' + str(x2).strip() + ' ' + str(y2).strip() + ' '
            # print(prediction_string)
    prediction_strings.append(prediction_string)
    file_names.append(image_id)
submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv('./submission_yolov5x_img640.csv', index=None)
