import cv2
import numpy as np
import torch
import torchvision
import utils
import model

from torchvision.transforms import Normalize
from PIL import Image

normalized_transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def process_inputs(image, max_selections=300, section_size=(224, 224)):
    images = []
    boxes = []
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()[:max_selections]
    rects = np.array([[x, y, x+w, y+h] for x, y, w, h in rects])

    for rect in rects:
        x1, y1, x2, y2 = rect
        img_section = image[y1: y2, x1: x2]
        img_section = cv2.resize(img_section, section_size)
        images.append(img_section)
        boxes.append(rect)

    images = np.array(images, dtype=np.float32)
    images = torch.from_numpy(images)
    images = images.permute(0, 3, 1, 2)
    images = normalized_transform(images)

    return images, np.array(boxes)

def non_max_supression(boxes, scores, labels, threshold=0.5, iou_threshold=0.5):
    idxs = np.where(scores>threshold)
    boxes = boxes[idxs]
    scores = scores[idxs]
    labels = labels[idxs]
    idxs = np.argsort(scores)
    chosen_boxes = []
    chosen_boxes_scores = []
    chosen_boxes_labels = []

    while len(idxs):
        last = len(idxs) - 1
        chosen_idx = idxs[last]
        chosen_box = boxes[chosen_idx]
        chosen_box_score = scores[chosen_idx]
        chosen_box_label = labels[chosen_idx]
        chosen_boxes.append(chosen_box)
        chosen_boxes_scores.append(chosen_box_score)
        chosen_boxes_labels.append(chosen_box_label)
        idxs = np.delete(idxs, last)
        i = len(idxs)-1

        while i >= 0:
            idx = idxs[i]
            curr_box = boxes[idx]
            curr_box_score = scores[idx]
            curr_box_label = labels[idx]
            if (curr_box_label == chosen_box_label and
                utils.calculate_iou_score(curr_box, chosen_box) > iou_threshold):
                idxs = np.delete(idxs, i)
            i -= 1

    return chosen_boxes, chosen_boxes_scores, chosen_boxes_labels

def process_outputs(scores, boxes, threshold=0.5, iou_threshold=0.5):
    labels = np.argmax(scores, axis=1)
    probas = np.max(scores, axis=1)
    idxs = labels != 0
    boxes = boxes[idxs]
    probas = probas[idxs]
    labels = labels[idxs]

    assert len(probas) == len(boxes) == len(labels)
    final_boxes, final_boxes_scores, final_boxes_labels = non_max_supression(boxes, probas, labels, threshold, iou_threshold)

    return final_boxes, final_boxes_scores, final_boxes_labels

device = 'cuda'
resnet_backbone = torchvision.models.resnet50(weights='IMAGENET1K_V2')

for param in resnet_backbone.parameters():
    param.requires_grad = False

model = model.build_model(backbone=resnet_backbone, num_classes=9)
model.to(device)
model.load_state_dict(torch.load('runs/rcnn/train/train.pt'))

image = np.array(Image.open('test_images/1.jpg'))
prep_val_images, prep_val_boxes = process_inputs(image)

model.eval()
with torch.no_grad():
    output = model(prep_val_images.to(device))

# Postprocess output from model
scores = torch.softmax(output, dim=1).cpu().numpy()
boxes, boxes_scores, boxes_labels = process_outputs(scores, prep_val_boxes, threshold=0.5, iou_threshold=0.5)

annotated_image = utils.draw_boxes(image, boxes, boxes_scores, boxes_labels)
cv2.imshow("R-CNN Inference", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)