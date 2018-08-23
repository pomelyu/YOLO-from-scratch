import numpy as np

from .io import load_anchors

def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x1 >= x2 or y1 >= y2:
        return 0

    area_interect = area([x1, y1, x2, y2])

    return area_interect / (area(box1) + area(box2) - area_interect)

def choose_anchor(w, h, anchors):
    best_anchor = -1
    best_iou = 0
    for i, anchor in enumerate(anchors):
        score = iou([0, 0, w, h], [0, 0, anchor[0], anchor[1]])
        if score > best_iou:
            best_anchor = i
            best_iou = score

    return best_anchor

