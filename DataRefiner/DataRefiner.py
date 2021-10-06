'''
이 코드는 다음 기능을 포함합니다.
 - Class imbalance를 줄이기 위해 각 클래스별 데이터 양을 균등하게 조정해 train, validation 데이터를 분할합니다.
 - Detector가 분류하기 어려운 크기(factor_minSize)를 제거합니다.
 - Train, Validation 데이터를 일정 비율(factor_trainValRatio)로 분리합니다.
 - COCO, Darknet 포맷으로 Annotation 데이터를 저장합니다.

    - COCO 포맷의 경우 다음과 같습니다.
        - 학습 타겟을 k-fold로 분할한 다음, train, validation 데이터로 나누어 저장합니다.

    - Darknet 포맷의 경우 다음과 같습니다.
        - 학습 타겟을 k-fold로 분할한 다음, train, validation 데이터로 나누어 저장합니다.
        - 이미지마다 bbox를 기술하는 txt 파일이 필요합니다. 이를 위해 자동으로 Annotation을 darknet 포맷으로 변환하고 이미지폴더에 txt파일을 생성합니다.
        - 클래스 이름을 기술하는 .names 파일을 생성합니다.

작성자 JiSeong Kim
최초 작성일 2021-10-01
최종 수정일 2021-10-06
'''
import json
import os
import random
import copy
import itertools


def main():
    # Path
    sourceAnnotation = 'C:/Dataset/Detection/dataset/train.json'  # 원시 train json 파일 경로를 지정해주세요
    sourceImageDir = 'C:/Dataset/Detection/dataset/train'  # train 이미지폴더를 지정해주세요
    targetDir = './Annotations'  # 새로운 dataset 파일이 저장될 위치입니다.
    # Factors
    factor_minSize = 20  # 제거할 Annotation의 width, 또는 height가 이 factor 이하라면 제거됩니다.
    shuffle = True  # 데이터셋 분할 시 Shuffle할 것인가에 대한 여부입니다.
    shuffle_seed = 7  # 랜덤시드를 설정합니다.
    kfold = 5  # k fold 의 k값을 지정합니다. 만약 사용 하고싶지 않은 경우, 5를 지정하시면 4:1 비율로 분리가 되기 때문에 아무 fold나 선택해 사용하시면 됩니다.

    # Tasks
    annotations = LoadAnnotations(sourceAnnotation)
    annotations = Preprocessing(annotations, factor_minSize=factor_minSize)
    folds = Split(annotations, k=kfold, shuffle=shuffle, shuffle_seed=shuffle_seed)
    Save2COCO(annotations, folds, targetDir)
    Save2Darknet(annotations, folds, targetDir, sourceImageDir)

'''
라운드로빈 방식으로 데이터를 분할합니다.
이는 k-fold 를 위해 구현되었습니다.
'''
def RoundRobinSublists(l, n=4):
    lists = [[] for _ in range(n)]
    i = 0
    for elem in l:
        lists[i].append(elem)
        i = (i + 1) % n
    return lists

'''
지정한 디렉터리가 없다면 디렉터리를 생성합니다.
만약 트리구조에서 중간루트가 없다면 중간루트도 자동으로 함께 생성합니다.
'''
def CreateDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


'''
COCO type 데이터셋으로부터 필수 정보만 추출합니다.
필수 정보의 기준은 다른 Annotation 포맷으로 변환되기 위한 최소한의 정보입니다.
현재 타겟 포멧은 다음과 같으며, 상황에 따라 더 추가될 수 있습니다.
 - COCO
 - Darknet
'''
def LoadAnnotations(sourceAnnotation):
    # Annotation load
    with open(sourceAnnotation, 'r') as f:
        json_origin = json.load(f)

    # Extract data for general transform
    annotations = {"images":[], "categories":[]}
    for image_origin in json_origin["images"]:
        image = {
            "width" : image_origin["width"],
            "height" : image_origin["height"],
            "file_name" : image_origin["file_name"].split('/')[1],
            "id": image_origin["id"],
            "bboxes":[],
            "primary":999,
        }
        annotations["images"].append(image)

    for category_origin in json_origin["categories"]:
        category = {
            "id" : category_origin["id"],
            "name" : category_origin["name"],
        }
        annotations["categories"].append(category)

    for bbox_origin in json_origin["annotations"]:
        bbox = {
            "image_id" : bbox_origin["image_id"],
            "category_id" : bbox_origin["category_id"],
            "area": bbox_origin["area"],
            "bbox" : bbox_origin["bbox"],
            "id" : bbox_origin["id"],
        }
        annotations["images"][bbox["image_id"]]["bboxes"].append(bbox)

    return annotations

'''
 - Detector가 분류하기 어려운 크기(factor_minSize)를 제거합니다.
'''
def Preprocessing(annotations, factor_minSize=20):
    print("전처리 전 Annotation 수 :", len([bboxes for bboxes in [image["bboxes"] for image in annotations["images"]]]))
    # Preprocessing the annotation
    for image in annotations["images"]:
        for bbox in image["bboxes"][:]:
            _, _, w, h = bbox["bbox"]
            if min(w, h) < factor_minSize:
                image["bboxes"].remove(bbox)
    print("전처리 후 Annotation 수 :", len([bboxes for bboxes in [image["bboxes"] for image in annotations["images"]]]))
    return annotations


# Remapping in order of importance
def Split(annotations, k=5, shuffle=True, shuffle_seed=7):
    # Suffle
    if shuffle:
        random.seed(shuffle_seed)
        random.shuffle(annotations["images"])

    # Define primary class
    mapper = {8: 0,  # Battery (159)
              9: 1,  # Clothing (468)
              2: 2,  # Paper pack (897)
              3: 3,  # Metal (936)
              4: 4,  # Glass (982)
              6: 5,  # Styrofoam (1,263)
              5: 6,  # Plastic (2,943)
              0: 7,  # General trash (3,996)
              7: 8,  # Plastic bag (5,178)
              1: 9,  # Paper (6,352)
              }
    for image in annotations["images"]:
        for bbox in image["bboxes"]:
            image["primary"] = min(image["primary"], mapper[bbox["category_id"]])

    images_byClass = {}
    for image in annotations["images"]:
        if images_byClass.get(image["primary"]) is None:
            images_byClass[image["primary"]] = []
        images_byClass[image["primary"]].append(image)

    # Split by K-Fold
    folds = [{"train":[], "valid":[]} for _ in range(k)]
    for image_class, images in images_byClass.items():
        fold = RoundRobinSublists(images, k)
        for idx in range(k):
            folds[idx]["train"] += list(itertools.chain(*fold[:idx])) + list(itertools.chain(*fold[idx+1:]))
            folds[idx]["valid"] += fold[idx]

    # View data distribution
    for idx in range(k):
        train_images = folds[idx]["train"]
        valid_images = folds[idx]["valid"]

        counter_train = [0 for i in range(10)]
        for image in train_images:
            for bbox in image["bboxes"]:
                counter_train[bbox["category_id"]] += 1

        counter_valid = [0 for i in range(10)]
        for image in valid_images:
            for bbox in image["bboxes"]:
                counter_valid[bbox["category_id"]] += 1

        counter_ratio = {}
        for idx in range(10):
            counter_ratio[idx] = counter_train[idx] / (counter_train[idx] + counter_valid[idx])

        for name, stage in zip(["train", "valid", "ratio"], [counter_train, counter_valid, counter_ratio]):
            print(name, end='   ')
            for idx in range(10):
                print(idx, ":", stage[idx], end=' ')
            print()

    return folds

'''
Annotation 정보를 Darknet format으로 저장합니다.
'''
def Save2Darknet(annotations, folds, targetDir, sourceImageDir):
    targetDir = os.path.join(targetDir, 'darknet')
    CreateDirectory(targetDir)

    with open(os.path.join(targetDir, "custom.names"), 'w') as f:
        for category in annotations["categories"]:
            f.write(category['name']+'\n')

    for image in annotations["images"]:
        path = os.path.join(sourceImageDir, image["file_name"].split('.')[0] + ".txt")
        image_width, image_height = image["width"], image["height"]
        dw, dh = 1. / image_width, 1. / image_height

        with open(path, 'w') as f:
            for bbox in image["bboxes"]:
                x, y, w, h = bbox["bbox"]
                cx = x + (w / 2)
                cy = y + (h / 2)

                rcx = round(cx * dw, 3)
                rw = round(w * dw, 3)
                rcy = round(cy * dh, 3)
                rh = round(h * dh, 3)

                f.write(str(bbox["category_id"]) + " " + str(rcx) + " " + str(rcy) + " " + str(rw) + " " + str(rh) + "\n")

    for idx, fold in enumerate(folds):
        with open(os.path.join(targetDir, "fold_"+str(idx+1)+"_train.txt"), 'w') as f:
            train_images = '\n'.join([os.path.join(sourceImageDir, image["file_name"]) for image in fold["train"]])
            f.write(train_images)

        with open(os.path.join(targetDir, "fold_"+str(idx+1)+"_valid.txt"), 'w') as f:
            train_images = '\n'.join([os.path.join(sourceImageDir, image["file_name"]) for image in fold["valid"]])
            f.write(train_images)

'''
Annotation 정보를 COCO format으로 저장합니다.
'''
def Save2COCO(annotations, folds, targetDir):
    targetDir = os.path.join(targetDir, 'coco')
    CreateDirectory(targetDir)
    for idx, fold in enumerate(folds):

        fold = copy.deepcopy(fold)
        annotations_train = copy.deepcopy(annotations)
        annotations_train["annotations"] = list(itertools.chain(*[bboxes for bboxes in [image["bboxes"] for image in fold["train"]]]))
        annotations_train["images"] = [image for image in fold["train"]]

        annotations_valid = copy.deepcopy(annotations)
        annotations_valid["annotations"] = list(itertools.chain(*[bboxes for bboxes in [image["bboxes"] for image in fold["valid"]]]))
        annotations_valid["images"] = [image for image in fold["valid"]]

        for image in annotations_train["images"]:
            image.pop('bboxes')
        for image in annotations_valid["images"]:
            image.pop('bboxes')

        # Save json file
        with open(os.path.join(targetDir, "fold_"+str(idx+1)+"_train.json"), 'w') as f:
            json.dump(annotations_train, f, indent=2)

        with open(os.path.join(targetDir, "fold_"+str(idx+1)+"_valid.json"), 'w') as f:
            json.dump(annotations_valid, f, indent=2)



if __name__ == '__main__':
    main()