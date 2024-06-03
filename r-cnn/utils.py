import os
import sys
import cv2
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

def convert_yolo_annotation(image, label_file):
    image_width = image.shape[1]
    image_height = image.shape[0]

    annotation = {}
    annotation['boxes'] = []
    annotation['classes'] = []
    with open(label_file, 'r') as file:
        for line in file:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            data = line_stripped.split(' ')
            
            x, y, w, h = float(data[1]), float(data[2]), float(data[3]), float(data[4])
            
            box_width = w * image_width
            box_height = h * image_height

            x_min = int(x * image_width - (box_width / 2))
            y_min = int(y * image_height - (box_height / 2))
            x_max = int(x * image_width + (box_width / 2))
            y_max = int(y * image_height + (box_height / 2))

            annotation['classes'].append(int(data[0]))
            annotation['boxes'].append((x_min, y_min, x_max, y_max))

    return annotation

def calculate_iou_score(box_1, box_2):
    '''
        box_1 = single of ground truth bounding boxes
        box_2 = single of predicted bounded boxes
    '''
    box_1_x1 = box_1[0]
    box_1_y1 = box_1[1]
    box_1_x2 = box_1[2]
    box_1_y2 = box_1[3]

    box_2_x1 = box_2[0]
    box_2_y1 = box_2[1]
    box_2_x2 = box_2[2]
    box_2_y2 = box_2[3]

    x1 = np.maximum(box_1_x1, box_2_x1)
    y1 = np.maximum(box_1_y1, box_2_y1)
    x2 = np.minimum(box_1_x2, box_2_x2)
    y2 = np.minimum(box_1_y2, box_2_y2)

    area_of_intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_box_1 = (box_1_x2 - box_1_x1 + 1) * (box_1_y2 - box_1_y1 + 1)
    area_box_2 = (box_2_x2 - box_2_x1 + 1) * (box_2_y2 - box_2_y1 + 1)
    area_of_union = area_box_1 + area_box_2 - area_of_intersection

    return area_of_intersection / float(area_of_union)

def process_data_for_rcnn(image, rects, annotations, iou_threshold, max_boxes):
    true_classes = []
    image_sections = []
    true_count = 0
    false_count = 0
    for i in range(len(annotations['boxes'])):
        label = annotations['classes'][i]
        box = annotations['boxes'][i]
        for rect in rects:
            iou_score = calculate_iou_score(rect, box)
            if iou_score > iou_threshold:
                if true_count < max_boxes // 2:
                    true_classes.append(label)
                    x1, y1, x2, y2 = rect
                    img_section = image[y1: y2, x1: x2]
                    image_sections.append(img_section)
                    true_count += 1
            else:
                if false_count < max_boxes // 2:
                    true_classes.append(0)
                    x1, y1, x2, y2 = rect
                    img_section = image[y1: y2, x1: x2]
                    image_sections.append(img_section)
                    false_count += 1

    return image_sections, true_classes

def draw_boxes(img, boxes, scores, labels):
    class_map = ['car', 'motorcycle', 'jeepney', 'bus', 'tricycle', 'van', 'truck', 'taxi', 'modern_jeepney']
    nums = len(boxes)

    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2])).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4])).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (240, 10, 150), 2)
        label = int(labels[i])
        label_txt = class_map[label]
        img = cv2.putText(
            img,
            "{} {:.2f}".format(label_txt, scores[i]),
            x1y1,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            2,
        )

    return img