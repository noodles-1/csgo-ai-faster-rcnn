import os
import cv2
import pickle
import numpy as np
import utils

from multiprocessing import Pool
from tqdm import tqdm

max_iou_threshold = 0.6
max_boxes = 50
max_selections = 1000
train_path = '../datasets/ph-vehicles/train'
val_path = '../datasets/ph-vehicles/val'
processed_data_save_path_train = "data/train/rcnn"
processed_data_save_path_val = "data/val/rcnn"
os.makedirs(processed_data_save_path_train, exist_ok=True)
os.makedirs(processed_data_save_path_val, exist_ok=True)

def process_image_annot(args):
    image, annotations = args
    image = np.array(image)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()[:max_selections]
    rects = np.array([[x, y, x+w, y+h] for x, y, w, h in rects])
    return utils.process_data_for_rcnn(image, rects, annotations, max_iou_threshold, max_boxes)

def save_processed_data(all_images, all_labels, processed_data_save_path):
    for idx, (image, label) in enumerate(zip(all_images, all_labels)):
        with open(os.path.join(processed_data_save_path, f"img_{idx}.pkl"), "wb") as pkl:
            pickle.dump({"image": image, "label": label}, pkl)

def main():
    all_images = []
    all_labels = []

    if len(os.listdir(processed_data_save_path_train)) < 80000:
        train_data = []

        for image_file in os.listdir(f'{train_path}/images'):
            image = cv2.imread(f'{train_path}/images/{image_file}')
            image_name = image_file[:len(image_file) - 4]
            label_file = f'{train_path}/labels-rcnn/{image_name}.txt'
            annotations = utils.convert_yolo_annotation(image, label_file)
            train_data.append((image, annotations))
        
        with Pool(processes=4) as pool:
            args = [(image, annot) for image, annot in train_data]
            results = list(tqdm(pool.imap(process_image_annot, args), total=len(args)))

        for images, classes in results:
            all_images += images
            all_labels += classes

        save_processed_data(all_images, all_labels, processed_data_save_path_train)
    else:
        print("Data Already Prepared.")

    all_images = []
    all_labels = []
    if len(os.listdir(processed_data_save_path_val)) < 80000:
        val_data = []
        for image_file in os.listdir(f'{val_path}/images'):
            image = cv2.imread(f'{train_path}/images/{image_file}')
            image_name = image_file[:len(image_file) - 4]
            label_file = f'{val_path}/labels-rcnn/{image_name}.txt'
            annotations = utils.convert_yolo_annotation(image, label_file)
            val_data.append((image, annotations))
  
        with Pool(processes=4) as pool:
            args = [(image, annot) for image, annot in val_data]
            results = list(tqdm(pool.imap(process_image_annot, args), total=len(args)))
        for images, classes in results:
            all_images += images
            all_labels += classes
        save_processed_data(all_images, all_labels, processed_data_save_path_val)
    else:
        print("Data Already Prepared.")

if __name__ == '__main__':
    main()