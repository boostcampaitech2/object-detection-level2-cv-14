'''
이 코드는 다음 기능을 포함합니다.
 - Detector가 분류하기 어려운 크기(factor_minSize)를 제거합니다.
 - Train, Validation 데이터를 일정 비율(factor_trainValRatio)로 분리합니다.

작성자 JiSeong Kim 2021-10-01
'''
import json
import os
import random

# Factors
annotation_source = 'C:/Dataset/Detection/dataset/train.json' # 원시 annotation json 파일 경로를 지정해주세요
annotation_targetDir = 'C:/Dataset/Detection/dataset/' # 새로운 annotation json 파일이 저장될 위치입니다.
factor_minSize = 20 # 제거할 Annotation의 width, 또는 height가 이 factor 이하라면 제거됩니다.
factor_trainRatio = 0.8 # train data의 비율을 의미하며, validation data는 1-factor 값으로 결정됩니다.
shuffle = True # 데이터셋 분할 시 Shuffle할 것인가에 대한 여부입니다.
seed = 7 # 랜덤시드를 설정합니다.

# Annotation Load
with open(annotation_source, 'r') as f:
    json_origin = json.load(f)

# Preprocessing the annotation
print("전처리 전 Annotation 수 :", len(json_origin["annotations"]))
annotations_edited = []
for annotation in json_origin["annotations"]:
    _, _, w, h = annotation["bbox"]
    if min(w, h) < factor_minSize:
        continue

    annotations_edited.append(annotation)

json_origin["annotations"] = annotations_edited
print("전처리 후 Annotation 수 :", len(json_origin["annotations"]))

# Define primary class order
mapper_origin = {
    "General trash":0,
    "Paper":1,
    "Paper pack":2,
    "Metal":3,
    "Glass":4,
    "Plastic":5,
    "Styrofoam":6,
    "Plastic bag":7,
    "Battery":8,
    "Clothing":9
}
mapper_refine = {
    mapper_origin["Battery"]:0, # 159
    mapper_origin["Clothing"]:1, # 468
    mapper_origin["Paper pack"]:2, # 897
    mapper_origin["Metal"]:3, # 936
    mapper_origin["Glass"]:4, # 982
    mapper_origin["Styrofoam"]:5, # 1,263
    mapper_origin["Plastic"]:6, # 2943
    mapper_origin["General trash"]:7, # 3,996
    mapper_origin["Plastic bag"]:8, # 5,178
    mapper_origin["Paper"]:9 # 6,352
}

# Mapping image and class relationships
images_byId = {}
for annotation in json_origin["annotations"]:
    if images_byId.get(annotation["image_id"]) is None:
        images_byId[annotation["image_id"]] = mapper_refine[annotation["category_id"]]
        continue
    images_byId[annotation["image_id"]] = min(images_byId[annotation["image_id"]], mapper_refine[annotation["category_id"]])

images_byClass = {}
for image_id, image_class in images_byId.items():
    if images_byClass.get(image_class) is None:
        images_byClass[image_class] = [image_id]
        continue
    images_byClass[image_class].append(image_id)

train_images, valid_images = [], []

random.seed(seed)
for images in images_byClass.values():
    fivot = int(len(images)*factor_trainRatio)
    if shuffle:
        random.shuffle(images)
    train_images += images[:fivot]
    valid_images += images[fivot:]

# Make each dataset
json_train = {"info":json_origin["info"], "license":json_origin["licenses"], "categories":json_origin["categories"], "images":[], "annotations":[]}
json_valid = {"info":json_origin["info"], "license":json_origin["licenses"], "categories":json_origin["categories"], "images":[], "annotations":[]}

for image in json_origin["images"]:
    if image["id"] in train_images:
        json_train["images"].append(image)
    elif image["id"] in valid_images:
        json_valid["images"].append(image)

for annotation in json_origin["annotations"]:
    if annotation["image_id"] in train_images:
        json_train["annotations"].append(annotation)
    elif annotation["image_id"] in valid_images:
        json_valid["annotations"].append(annotation)

# Save json file
with open(os.path.join(annotation_targetDir, "new_train.json"),'w') as f:
    json.dump(json_train, f, indent=2)

with open(os.path.join(annotation_targetDir, "new_valid.json"),'w') as f:
    json.dump(json_valid, f, indent=2)


# View data distribution
counter_origin = [0 for i in range(10)]
for annotation in json_origin["annotations"]:
    counter_origin[annotation["category_id"]] += 1

counter_train = [0 for i in range(10)]
for annotation in json_train["annotations"]:
    counter_train[annotation["category_id"]] += 1

counter_valid = [0 for i in range(10)]
for annotation in json_valid["annotations"]:
    counter_valid[annotation["category_id"]] += 1

counter_ratio = {}
for idx in range(10):
    counter_ratio[idx] = counter_train[idx]/counter_origin[idx]

for name,stage in zip(["origin","train","valid","ratio"],[counter_origin, counter_train, counter_valid, counter_ratio]):
    print(name,end='   ')
    for idx in range(10):
        print(idx,":",stage[idx],end=' ')
    print()