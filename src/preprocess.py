import numpy as np
import os
from skimage.transform import resize
from .utils import choose_anchor
from .constants import MODEL_DIM, GRID_SIZE, YOLO1_CLASS

class ResizeTransform():
    def __init__(self, target_w=MODEL_DIM, target_h=MODEL_DIM):
        self.target_w = target_w
        self.target_h = target_h

    def transform(self, image, bbox=None):
        h, w = image.shape[:2]
        dim_h, dim_w = self.target_h, self.target_w
        ratio = min(dim_w/w, dim_h/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = resize(image, (new_h, new_w), mode="constant", preserve_range=True)
        resized = resized.astype(np.uint8)

        padded = np.zeros((dim_h, dim_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        if type(bbox) == type(None):
            return padded

        br_h = new_h / dim_h
        br_w = new_w / dim_w

        res_bbox = np.copy(bbox)
        res_bbox[:, 0] = bbox[:, 0] * br_w
        res_bbox[:, 1] = bbox[:, 1] * br_h
        res_bbox[:, 2] = bbox[:, 2] * br_w
        res_bbox[:, 3] = bbox[:, 3] * br_h

        return padded, res_bbox

    def reverse_transform_bbox(self, bbox, ori_shape):
        h, w = ori_shape[:2]
        dim_h, dim_w = self.target_h, self.target_w
        ratio = min(dim_w/w, dim_h/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        br_h = new_h / dim_h
        br_w = new_w / dim_w

        res_bbox = np.copy(bbox)
        res_bbox[:, 0] = bbox[:, 0] / br_w
        res_bbox[:, 1] = bbox[:, 1] / br_h
        res_bbox[:, 2] = bbox[:, 2] / br_w
        res_bbox[:, 3] = bbox[:, 3] / br_h

        return res_bbox

def bbox_2_bbox_map(bboxes, anchors, grid_size=GRID_SIZE, classes=YOLO1_CLASS):
    num_box = len(anchors)
    box_dim = 5 + len(classes)

    res = np.zeros((grid_size, grid_size, num_box, box_dim))
    cell_dim = 1 / grid_size

    for bbox in bboxes:
        class_idx = int(bbox[0])
        idx_x = int(bbox[1] / cell_dim)
        idx_y = int(bbox[2] / cell_dim)

        anchor_id = choose_anchor(bbox[3], bbox[4], anchors)
        res[idx_y, idx_x, int(anchor_id), :5] = np.array([ 1, bbox[1], bbox[2], bbox[3], bbox[4] ])
        res[idx_y, idx_x, int(anchor_id), 5 + class_idx] = 1

    return res

# def preprocess_image(image, dim=MODEL_DIM):
#     h, w, _ = image.shape
#     ratio = min(dim / w, dim / h)
#     new_w = int(w * ratio)
#     new_h = int(h * ratio)
#     resized = resize(image, (new_h, new_w), mode="constant", preserve_range=True)
#     resized = resized.astype(np.uint8)
    
#     padded = np.zeros((dim, dim, 3))
#     pad_h = (dim - new_h) // 2
#     pad_w = (dim - new_w) // 2
#     padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = resized

#     return padded

# def preprocess_label(labels, image_shape, anchors, dim=MODEL_DIM, grid_size=GRID_SIZE, classes=YOLO1_CLASS):
#     num_box = len(anchors)
#     box_dim = 5 + len(classes)

#     h, w = image_shape
#     ratio = min(dim / w, dim / h)
#     offset_h = (dim - h * ratio ) / 2
#     offset_w = (dim - w * ratio ) / 2

#     res = np.zeros((grid_size, grid_size, num_box, box_dim))
#     cell_dim = dim / grid_size

#     for label in labels:
#         class_idx = int(label[0])
#         cx = offset_w + label[1] * w * ratio
#         cy = offset_h + label[2] * h * ratio
#         bw = label[3] * w * ratio / dim
#         bh = label[4] * h * ratio / dim
#         idx_x = int(cx // cell_dim)
#         idx_y = int(cy // cell_dim)

#         anchor_id = choose_anchor(bw, bh, anchors)
#         res[idx_y, idx_x, int(anchor_id), :5] = np.array([ 1, cx/dim, cy/dim, bw, bh ])
#         res[idx_y, idx_x, int(anchor_id), 5 + class_idx] = 1

#     return res

# def preprocess_all(image_dir, label_dir, image_out, label_out, anchors, \
#     dim=MODEL_DIM, grid_size=GRID_SIZE, classes=YOLO1_CLASS):

#     generator = image_label_generator(image_dir, label_dir)
    

#     while True:
#         try:
#             file_name, image, label = next(generator)
#         except StopIteration:
#             break

#         resized = preprocess_image(image, dim)
#         bbox_map = preprocess_label(label, image.shape[:2], anchors, dim, grid_size, classes)

#         save_image(os.path.join(image_out, "{}.jpg".format(file_name)), resized)
#         save_npy(os.path.join(label_out, "{}.npy".format(file_name)), bbox_map)

# def restore_label(label, image_shape, dim=MODEL_DIM):
#     res = np.array(label)
#     h, w = image_shape
#     ratio = min(dim / w, dim / h)
#     offset_w = (dim - w * ratio ) / 2
#     offset_h = (dim - h * ratio ) / 2
#     divide_w = w * ratio
#     divide_h = h * ratio

#     if len(res) != 0:
#         res[:, 2] = (res[:, 2] * dim - offset_w) / divide_w
#         res[:, 3] = (res[:, 3] * dim - offset_h) / divide_h
#         res[:, 4] = res[:, 4] * dim / divide_w
#         res[:, 5] = res[:, 5] * dim / divide_h

#     return res
