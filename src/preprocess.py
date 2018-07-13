import numpy as np
import os
from skimage import io
from skimage import transform

from .constants import MODEL_DIM, GRID_SIZE, CLASS_NAME, NUM_BOX, NUM_CLASS, YOLO1_CLASS, YOLO2_CLASS, GAMMA_MIN, GAMMA_MAX
from .utils import load_image, load_labels, isExtension, yolo1_to_yolo_2
from .image_transform import flip_image_horizontal, adjust_gamma

def scale_image_with_padding(image, label, model_dim):
    """ Resize image to required dimension with necessary padding and adjust the label coordinates

    Arguments:
        image: ndarray
        label: objects label
        model_dim: the required dimension

    Returns:
        padded: ndarray, resized and padded image
        label:
    """
    h, w, _ = image.shape
    ratio = min(model_dim / w, model_dim / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = transform.resize(image, (new_h, new_w), mode="constant", anti_aliasing=True)
    
    padded = np.zeros((model_dim, model_dim, 3))
    pad_h = (model_dim - new_h) // 2
    pad_w = (model_dim - new_w) // 2
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = resized
    
    label[:, 1] = (label[:, 1] * new_w + pad_w) / model_dim
    label[:, 2] = (label[:, 2] * new_h + pad_h) / model_dim
    label[:, 3] = (label[:, 3] * new_w) / model_dim
    label[:, 4] = (label[:, 4] * new_h) / model_dim

    return padded, label

def generate_bboxs(labels, model_dim, grid_size, num_box, num_class):
    """ Divide image into grids and generate target value for YOLO training

    Arguments:
        labels: labels from text file
        model_dim: the required dimension
        grid_size: the size of grids
        num_box: maximum bounding box(objects) could be described by a cell
        num_class: number of object class

    Returns:
        bboxs: the numpy array with (grid_size, grid_size, num_box, 5+num_class)
    """
    obj_in_cells = {}
    for label in labels:
        # align the dimension
        bx, by, bw, bh = label[1], label[2], label[3], label[4]

        if CLASS_NAME == YOLO1_CLASS:
            class_index = label[0]
        elif CLASS_NAME == YOLO2_CLASS:
            class_index = yolo1_to_yolo_2(label[0])

        # calculate the cell which this bbox belong to
        grid_x = bx * grid_size
        grid_y = by * grid_size

        cell_idx = (int(grid_x), int(grid_y))
        
        obj = (class_index, grid_x - cell_idx[0], grid_y - cell_idx[1], bw, bh)
        if cell_idx not in obj_in_cells:
            obj_in_cells[cell_idx] = [obj]
        else:
            obj_in_cells[cell_idx].append(obj)

    box_size = 5 + num_class
    bboxs = np.zeros((grid_size, grid_size, num_box, box_size))
    for (i, j), objs in obj_in_cells.items():
        for idx, bbox in enumerate(objs):
            if idx > 1: break

            bboxs[i][j][idx][0] = 1
            bboxs[i][j][idx][1] = bbox[1]
            bboxs[i][j][idx][2] = bbox[2]
            bboxs[i][j][idx][3] = bbox[3]
            bboxs[i][j][idx][4] = bbox[4]
            bboxs[i][j][idx][5 + int(bbox[0])] = 1

    return bboxs



def preprocess_data(imgs_dir, labels_dir, imgs_out, label_out, arg_factor=1):
    labels = [label for label in os.listdir(labels_dir) if isExtension(label, ".txt")]

    for idx in range(len(labels)):
        fil_name = labels[idx].replace(".txt", "")
        image = load_image(os.path.join(imgs_dir, "{}.jpg".format(fil_name)))
        label = load_labels(os.path.join(labels_dir, "{}.txt".format(fil_name)))

        image, label = scale_image_with_padding(image, label, MODEL_DIM)

        # Without argumentation
        if arg_factor == 1:
            bboxs = generate_bboxs(label, MODEL_DIM, GRID_SIZE, NUM_BOX, NUM_CLASS)
            io.imsave(os.path.join(imgs_out, "pre_{}.jpg".format(fil_name)), image)
            np.save(os.path.join(label_out, "pre_{}.npy".format(fil_name)), bboxs)
        
        # With argumentation
        np.random.seed(idx)
        flip_val = np.random.permutation(arg_factor) > (arg_factor // 2)
        np.random.seed(idx+1)
        gamma_val = np.random.permutation(arg_factor) * ((GAMMA_MAX - GAMMA_MIN) / arg_factor) + GAMMA_MIN

        for i in range(arg_factor):
            arg_image, arg_label = argument_image(image, label, flip=flip_val[i], gamma=gamma_val[i])
            bboxs = generate_bboxs(arg_label, MODEL_DIM, GRID_SIZE, NUM_BOX, NUM_CLASS)

            io.imsave(os.path.join(imgs_out, "{}_{:0>2d}.jpg".format(fil_name, i)), arg_image)
            np.save(os.path.join(label_out, "{}_{:0>2d}.npy".format(fil_name, i)), bboxs)


def argument_image(image, label, seed=0, flip=False, gamma=1):
    arg_image = np.copy(image)
    arg_label = np.copy(label) 

    if flip:
        arg_image, arg_label = flip_image_horizontal(arg_image, arg_label)
    arg_image = adjust_gamma(arg_image, gamma)

    return arg_image, arg_label
