import json
import os
import pandas as pd

with open("result.txt", 'r') as f:
    json_origin = json.load(f)

prediction_strings = []
file_names = []

for annotations in json_origin:
    prediction_string = ''
    for bbox in annotations["objects"]:
        cls = bbox["class_id"]
        confidence = bbox["confidence"]
        cx = bbox["relative_coordinates"]["center_x"] * 1024
        cy = bbox["relative_coordinates"]["center_y"] * 1024
        w = bbox["relative_coordinates"]["width"] * 1024
        h = bbox["relative_coordinates"]["height"] * 1024
        x1 = max(cx - w/2, 0)
        y1 = max(cy - h/2, 0)
        x2 = min(cx + w/2, 1023)
        y2 = min(cy + h/2, 1023)

        prediction_string += str(cls) + ' ' + str(confidence) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' '
    prediction_strings.append(prediction_string)
    file_names.append('test/'+annotations["filename"].split('/')[-1])

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission.to_csv('./yolov4_darknet_submission.csv', index=None)