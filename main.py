"""
1.读取coco数据集的json文件
2.分析json文件，获取图片信息
3.分析json文件，获取标注信息
4.将图片和标注信息合并到一起，保存到txt文件中
5.统计分类信息，写入txt文件中
"""

import json
import os
from pycocotools.coco import COCO
import cv2
import random

# 使用环境变量或配置文件来设置路径
train_dir = os.getenv('TRAIN_DIR', "./dataset/coco/train")
val_dir = os.getenv('VAL_DIR', "./dataset/coco/val")
test_dir = os.getenv('TEST_DIR', "./dataset/coco/test")
train_json = os.getenv('TRAIN_JSON', "./dataset/coco/instances_train2017.json")
val_json = os.getenv('VAL_JSON', "./dataset/coco/instances_val2017.json")

def load_coco_json(json_file):
    try:
        coco = COCO(json_file)
    except Exception as e:
        print(f"Error loading JSON file: {json_file}. Error: {str(e)}")
        raise
    return coco

def save_info(coco, output_file, info_func):
    try:
        with open(output_file, 'w') as f:
            for img_id in coco.imgs.keys():
                f.write(info_func(coco, img_id))
    except Exception as e:
        print(f"Error saving info to {output_file}. Error: {str(e)}")

def save_image_info(coco, output_file):
    def image_info_func(coco, img_id):
        img_info = coco.imgs[img_id]
        return f"Image ID: {img_id}, File Name: {img_info['file_name']}, Height: {img_info['height']}, Width: {img_info['width']}\n"
    
    save_info(coco, output_file, image_info_func)

def save_annotation_info(coco, output_file):
    def annotation_info_func(coco, img_id):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        ann_info = "".join([f"Annotation ID: {ann['id']}, Image ID: {ann['image_id']}, Category ID: {ann['category_id']}, Bbox: {ann['bbox']}\n" for ann in coco.loadAnns(ann_ids)])
        return ann_info
    
    save_info(coco, output_file, annotation_info_func)

def convert_to_yolo_format(coco, output_dir):
    coco_images = coco.imgs
    for img_id in coco_images.keys():
        img_info = coco_images[img_id]
        file_name = img_info['file_name']
        h, w = img_info['height'], img_info['width']

        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        txt_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.txt')
        try:
            with open(txt_file, 'w') as f:
                for ann in annotations:
                    x, y, width, height = ann['bbox']
                    x_center = (x + width / 2) / w
                    y_center = (y + height / 2) / h
                    width = width / w
                    height = height / h
                    category_id = ann['category_id']
                    class_index = category_id_to_class_index(category_id)
                    f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        except Exception as e:
            print(f"Error saving YOLO format to {txt_file}. Error: {str(e)}")

def category_id_to_class_index(category_id):
    class_index = {1: 'class1', 2: 'class2', 3: 'class3'}
    return class_index.get(category_id, category_id)

def save_categories(json_file_path, output_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        categories = data.get('categories', [])

        with open(output_file_path, 'w') as f:
            for category in categories:
                category_id = category['id']
                category_name = category['name']
                f.write(f"{category_id}: {category_name}\n")
    except Exception as e:
        print(f"Error saving categories to {output_file_path}. Error: {str(e)}")

def save_class_labels(json_file_path, output_dir="./"):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    categories = data['categories']

    with open(os.path.join(output_dir, "categories.txt"), 'w') as out_file:
        for category in categories:
            out_file.write(f"{category['id']}: {category['name']}\n")

def format_conversion(work_json_file,output_dir):
    json_file = work_json_file
    coco = load_coco_json(json_file)
    
    save_class_labels(json_file, r"./")
    
    image_info_file = "image_info.txt"
    save_image_info(coco, image_info_file)

    annotation_info_file = "annotation_info.txt"
    save_annotation_info(coco, annotation_info_file)
    
    os.makedirs(output_dir, exist_ok=True)  # 安全地创建目录
    convert_to_yolo_format(coco, output_dir)


def verify_annotations(images_dir, labels_dir, output_dir, num_images=3):
    class_labels = {}
    labels_file_path = './categories.txt'
    
    with open(labels_file_path, 'r') as f:
        for line in f:
            class_index, label = line.strip().split(':')
            class_labels[int(class_index)] = label
    
    images_list = os.listdir(images_dir)
    num_images = min(len(images_list), num_images)

    selected_images = random.sample(images_list, num_images)
    os.makedirs(output_dir, exist_ok=True)

    for img_name in selected_images:
        image_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()  # 分割每一行数据
            if len(parts) < 5:
                print(f"Warning: Line '{line}' does not contain enough data.")
                continue
            # 检查类标签是否为数字，并从class_labels字典中获取正确的标签
            class_index = parts[0]
            if not class_index.isdigit():  # 检查是否为数字
                #print(f"Warning: Non-numeric class index found: {class_index}")
                continue

            x_center, y_center, box_width, box_height = map(float,  parts[1:])
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = class_labels[int(class_index)]
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        output_image_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_image_path, image)



if __name__ == "__main__":
    # format_conversion(r"D:\MyCode2023\coco2yolo\dataset\coco\instances_train2017.json",
    #                   r"D:\MyCode2023\coco2yolo\dataset\coco\train\labels")

    verify_annotations(r'D:\MyCode2023\coco2yolo\dataset\coco\train\images',
                       r'D:\MyCode2023\coco2yolo\dataset\coco\train\labels',
                       r'./output', num_images=5)
