import os
import cv2

'''
이 코드는 모델이 Test 데이터를 추론한 결과를 눈으로 확인하기 위한 Visualization 목적으로 제작하였습니다.
이 코드는 GUI 환경에서 작동되므로, 콘솔환경에서는 사용하실 수 없습니다.

Visualization 타겟은 다음과 같습니다.
 - Target Image
 - Bounding Box
 - Confidence
 - Class Name

이 코드는 다음 Dependency가 필요합니다.
 - opencv
  - 설치 방법
   - pip install opencv-python

이 코드는 다음 기능으로 프로세스를 컨트롤 할 수 있습니다.
- right Key : 인덱스를 1칸 앞으로 이동시킵니다.
- left Key : 인덱스를 1칸 뒤로 이동시킵니다.
- up Key : 인덱스를 10칸 앞으로 이동시킵니다.
- down Key : 인덱스를 10칸 뒤로 이동시킵니다.
- esc Key : 프로세스를 종료합니다.
- S Key : 현재 표시되는 시각화된 이미지를 저장합니다.

작성자 JiSeong Kim
최초 작성일 2021-10-07
'''

imageRoot = r'C:/Dataset/Detection/dataset/test' # Test 이미지가 있는 경로를 입력합니다.
annotationPath = r'C:/Dataset/submission_epoch_24.csv' # 모델에 의해 추론된 최종 제출 csv 파일 경로를 입력합니다.

def main():
    # 클래스 인덱스에 따른 클래스이름을 정의합니다.
    mapper_cls = {
        0: 'General trash',
        1: 'Paper',
        2: 'Paper pack',
        3: 'Metal',
        4: 'Glass',
        5: 'Plastic',
        6: 'Styrofoam',
        7: 'Plastic bag',
        8: 'Battery',
        9: 'Clothing'
    }

    # 클래스 인덱스에 따른 boundingBox 의 색상을 정의합니다.
    mapper_color = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (255, 0, 255),
        5: (0, 255, 255),
        6: (255, 255, 255),
        7: (128, 128, 0),
        8: (128, 0, 128),
        9: (0, 128, 128),
    }

    # csv파일을 읽어 이미지와 bounding box의 관계를 정의합니다.
    annotations = []
    with open(annotationPath, 'r') as f:
        lines = f.readlines()

        for line in lines[1:]:
            fileName = line.split(',')[1].split('/')[-1].strip()
            bboxes_raw = list_chunk(line.split(',')[0].strip().split(' '), 6)

            bboxes = []
            for box in bboxes_raw:
                if not box[0]:
                    continue

                id, confidence, x1, y1, x2, y2 = box
                bboxes.append([int(id), float(confidence), int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))])
            annotations.append({'fileName':fileName,'bboxes':bboxes})

    # 한 이미지에 표현된 boundingBox를 시각화합니다.
    idx = 0
    idx_max = len(annotations)
    while True:
        annotation = annotations[idx]
        imagePath = os.path.join(imageRoot, annotation["fileName"])
        frame = cv2.imread(imagePath, cv2.IMREAD_COLOR)

        # draw image name
        frame = cv2.putText(frame, f"{annotation['fileName']}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # draw bboxes
        for id, confidence, x1, y1, x2, y2 in annotation["bboxes"]:
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2), mapper_color[id], 2)
            frame = cv2.putText(frame, f"{int(confidence*100)}% {mapper_cls[id]}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  mapper_color[id], 1)

        # Visualize image
        cv2.imshow('frame', frame)

        # Weight keyboard event
        key = cv2.waitKeyEx(0)
        if key==0x270000: # right Key : 인덱스를 1칸 앞으로 이동시킵니다.
            idx = min(idx+1, idx_max)
        elif key==0x250000: # left Key : 인덱스를 1칸 뒤로 이동시킵니다.
            idx = max(idx-1, 0)
        elif key==0x260000: # up Key : 인덱스를 10칸 앞으로 이동시킵니다.
            idx = min(idx+10, idx_max)
        elif key==0x280000: # down Key : 인덱스를 10칸 뒤로 이동시킵니다.
            idx = max(idx-10, 0)
        elif key==0x1B: # esc Key : 프로세스를 종료합니다.
            break
        elif key==115: # S Key : 현재 표시되는 시각화된 이미지를 저장합니다.
            cv2.imwrite(annotation["fileName"], frame)

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

if __name__ == '__main__':
    main()