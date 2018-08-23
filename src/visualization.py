import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .constants import YOLO1_CLASS
from .io import load_image, load_label

def visualize_label(image_path, label_path, classes=YOLO1_CLASS):
    image = load_image(image_path)
    labels = load_label(label_path)
    
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_title(image_path)
    ax.imshow(image)
    
    h, w, _ = image.shape
    for label in labels:
        class_name = classes[int(label[0])]
        draw_label_rect(class_name, label[1] * w, label[2] * h, label[3] * w, label[4] * h, ax)

    plt.show()


def visualize_bbox_map(image, bbox_map, classes=YOLO1_CLASS):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.imshow(image)

    h, w, _ = image.shape
    bbox_idxs = np.where(bbox_map[..., 0] > 0)
    for iy, ix, ib in zip(bbox_idxs[0], bbox_idxs[1], bbox_idxs[2]):
        bbox = bbox_map[iy, ix, ib]
        class_idx = np.argmax(bbox[5:])
        class_name = classes[class_idx]
        draw_label_rect(class_name, bbox[1] * w, bbox[2] * h, bbox[3] * w, bbox[4] * h, ax)


def visualize_score_bbox(image, scores, boxes, classes, class_name=YOLO1_CLASS, color="blue"):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.imshow(image)
    
    model_dim, _, _ = image.shape
    
    for score, box, class_index in zip(scores, boxes, classes):
        x = box[0] * model_dim
        y = box[1] * model_dim
        w = (box[2] - box[0]) * model_dim
        h = (box[3] - box[1]) * model_dim
        
        name = "{}: {:.2}".format(class_name[class_index], score)
        
        ax.text(x, y, name, fontsize=12, horizontalalignment="left", 
            verticalalignment="top", backgroundcolor=color, color="white")
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor=color))


def visualize_score_label(image, labels, classes=YOLO1_CLASS):    
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.imshow(image)
    
    h, w, _ = image.shape
    for label in labels:
        class_name = "{}: {:.2}".format(classes[int(label[1])], label[0])
        draw_label_rect(class_name, label[2] * w, label[3] * h, label[4] * w, label[5] * h, ax)

    plt.show()


def draw_label_rect(name, cx, cy, w, h, ax, color="blue"):
    x = cx - (w / 2)
    y = cy - (h / 2)
    ax.text(x, y, name, fontsize=10, horizontalalignment="left", 
        verticalalignment="bottom", backgroundcolor=color, color="white", 
        bbox=dict(boxstyle='square, pad=0', ec='none'))
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor=color))
