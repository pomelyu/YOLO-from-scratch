import numpy as np
import os
from PIL import Image
from skimage import io

from .constants import MODEL_DIM, GRID_SIZE, CLASS_NAME, NUM_BOX, NUM_CLASS, YOLO1_CLASS, YOLO2_CLASS, GAMMA_MIN, GAMMA_MAX
from .utils import load_image, load_labels, isExtension, yolo1_to_yolo_2
from .image_transform import flip_image_horizontal, adjust_gamma

def preprocess_image(image, model_dim):
    """ Resize image to required dimension and add necessary padding

    Arguments:
        image: PIL image object
        model_dim: the required dimension
    """
    img_size = image.size
    ratio = min(model_dim / img_size[0], model_dim / img_size[1])
    new_w = int(img_size[0] * ratio)
    new_h = int(img_size[1] * ratio)
    resized = image.resize((new_w, new_h), Image.ANTIALIAS)

    padded = Image.new("RGB", (model_dim, model_dim))
    padded.paste(resized, box=((model_dim-new_w) // 2, (model_dim-new_h) // 2))
    
    return padded

def generate_bboxs(labels, image, model_dim, grid_size, num_box, num_class):
    """ Divide image into grids and generate target value for YOLO training

    Arguments:
        labels: labels from text file
        image: the original image
        model_dim: the required dimension
        grid_size: the size of grids
        num_box: maximum bounding box(objects) could be described by a cell
        num_class: number of object class

    Returns:
        bboxs: the numpy array with (grid_size, grid_size, num_box, 5+num_class)
    """
    img_size = image.size
    ratio = min(model_dim / img_size[0], model_dim / img_size[1])
    new_w = int(img_size[0] * ratio)
    new_h = int(img_size[1] * ratio)

    obj_in_cells = {}
    for label in labels:
        # align the dimension
        bx = (label[1] * new_w + (model_dim - new_w) // 2) / model_dim
        by = (label[2] * new_h + (model_dim - new_h) // 2) / model_dim
        bw = (label[3] * new_w) / model_dim
        bh = (label[4] * new_h) / model_dim

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
            bboxs[i][j][idx][5 + bbox[0]] = 1

    return bboxs


def preprocess_data(imgs_path, labels_path, sav_imgs_path, save_labels_path):
    labels = [label for label in os.listdir(labels_path) if isExtension(label, ".txt")]
    
    for i in range(len(labels)):
        image_name = labels[i].replace(".txt", ".jpg")
        image = load_image(os.path.join(imgs_path, image_name))
        preprocessed = preprocess_image(image, MODEL_DIM)
        
        image_name = "pre_" + image_name
        preprocessed.save(os.path.join(sav_imgs_path, image_name))
        
        label = load_labels(os.path.join(labels_path, labels[i]))
        bboxs = generate_bboxs(label, image, MODEL_DIM, GRID_SIZE, NUM_BOX, NUM_CLASS)
        
        bboxs_name = "pre_" + labels[i][:labels[i].rindex(".")]
        np.save(os.path.join(save_labels_path, bboxs_name), bboxs)
        
        
def argument_data(images_dir, labels_dir, images_out, labels_out, argument_size=5):
    files = np.array([
        label.replace(".npy", "") for label in os.listdir(labels_dir) if isExtension(label, ".npy")
    ])
    cwd = os.getcwd()
    for file in files:
        image = io.imread(os.path.join(cwd, images_dir, "{}.jpg".format(file)))
        bboxs = np.load(os.path.join(cwd, labels_dir, "{}.npy".format(file)))

        file_num = int(file[file.index("0"):])
        for i in range(argument_size):
            arg_image = np.copy(image)
            arg_bboxs = np.copy(bboxs)
            
            np.random.seed(i + file_num)
            rand_num = np.random.rand()
            
            if rand_num < 0.5:
                arg_image, arg_bboxs = flip_image_horizontal(arg_image, arg_bboxs)
                
            gamma = GAMMA_MIN + rand_num * (GAMMA_MAX - GAMMA_MIN)
            arg_image = adjust_gamma(arg_image, gamma)
            
            io.imsave(os.path.join(cwd, images_out, "{}_{}.jpg".format(file, i)), arg_image)
            np.save(os.path.join(cwd, labels_out, "{}_{}.npy".format(file, i)), arg_bboxs)
