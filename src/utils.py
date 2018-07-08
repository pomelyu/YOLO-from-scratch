import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from PIL import Image

from .constants import YOLO1_CLASS, YOLO2_CLASS, CLASS_NAME, NUM_CLASS, MODEL_DIM
from .image_transform import covert_to_VGG_input

def load_image(image_path):
    """ load image from file
    Arguments:
    image_path: relative path of the image

    Return:
    image: PIL image object
    """
    if os.path.isabs(image_path):
        path = image_path
    else:
        cwd = os.getcwd()
        path = os.path.join(cwd, image_path)
    img = Image.open(path)

    return img


def load_labels(label_path):
    """ load label information from file
    Arguments:
    label_path: relative path of the file preprcossed by darknet tool
        line example: classname x y w h (x, y, w, h is normalized)

    Return:
    labels: array of object information
        [[class_label, x, y, w, h], [class_label, x, y, w, h]]
    """
    if os.path.isabs(label_path):
        path = label_path
    else:
        cwd = os.getcwd()
        path = os.path.join(cwd, label_path)

    labels = []
    with open(path, 'r') as fin:
        for line in fin:
            obj = [ float(num) for num in line.split(" ") ]
            obj[0] = int(obj[0])
            labels.append(obj)

    return labels

def draw_label_rect(name, cx, cy, w, h, ax, color="blue"):
    x = cx - (w / 2)
    y = cy - (h / 2)
    ax.text(x, y, name, fontsize=12, horizontalalignment="left", 
        verticalalignment="top", backgroundcolor=color, color="white")
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor=color))


def visualize_label(image_path, label_path):
    """ display the bounding box and classes of objects in the image

    Arguments:
    img_path: relative path of the image
    label_path: relative path of the text for bounding box information
        line example: classname x y w h (x, y, w, h is normalized)

    Example:
    visualize_label("../data/VOC2007/images/000007.jpg", "../data/VOC2007/labels/000007.txt")

    """
    img = load_image(image_path)
    labels = load_labels(label_path)
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_title(image_path)
    ax.imshow(img)
    
    w, h = img.size
    for label in labels:
        class_name = YOLO1_CLASS[label[0]]
        draw_label_rect(class_name, label[1] * w, label[2] * h, label[3] * w, label[4] * h, ax)

    plt.show()

def draw_bboxs(image, bboxs, threshold=0, show_grid=True, show_center=True):
    """ display the bounding box from the YOLO bounding box representation

    Arguments:
        image: the preprocessed image
        bboxs: the numpy array with (grid_size, grid_size, num_box, 5+num_class)
        threshold: bounding box confidence threshold

    """
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.imshow(image)

    model_dim = image.size[0]
    grid_size = bboxs.shape[0]
    cell_dim = model_dim / grid_size

    if show_grid:
        for i in range(grid_size):
            ax.plot([0, model_dim], [i*cell_dim, i*cell_dim], color="white")
            ax.plot([i*cell_dim, i*cell_dim], [0, model_dim], color="white")

    for i in range(grid_size):
        for j in range(grid_size):
            for box_idx in [0, 1]:
                bbox = bboxs[i][j][box_idx]
            
                # Supress box with low confidence
                if bbox[0] <= threshold:
                    continue
                
                class_index= bbox[5: 5 + NUM_CLASS].argmax()
                class_name = CLASS_NAME[class_index]
                x = (i + bbox[1]) * cell_dim
                y = (j + bbox[2]) * cell_dim
                w = bbox[3] * model_dim
                h = bbox[4] * model_dim

                if show_center:
                    ax.annotate(class_name, xy=(x, y), color="white")
                    ax.plot(x, y, 'o')

                draw_label_rect(class_name, x, y, w, h, ax)
                
    
    ax.set_xlim(0, model_dim)
    ax.set_ylim(model_dim, 0)
    plt.show()

def draw_score_bbox(image, scores, boxes, classes, color="blue"):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.imshow(image)
    
    model_dim, _, _ = image.shape
    
    for score, box, class_index in zip(scores, boxes, classes):
        x = box[0] * model_dim
        y = box[1] * model_dim
        w = (box[2] - box[0]) * model_dim
        h = (box[3] - box[1]) * model_dim
        
        name = "{}: {:.2}".format(CLASS_NAME[class_index], score)
        
        ax.text(x, y, name, fontsize=12, horizontalalignment="left", 
            verticalalignment="top", backgroundcolor=color, color="white")
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor=color))


def isExtension(file_path, extension):
    file_ext = os.path.splitext(file_path)[-1].lower()
    return file_ext == extension.lower()

def yolo1_to_yolo_2(yolo1_class_index):
    name = YOLO1_CLASS[yolo1_class_index]
    return YOLO2_CLASS.index(name)

def convert_yolo_format(image, labels):
    w, h = image.size
    converted = labels.copy()
    for label in converted:
        label[0] = CLASS_NAME[label[0]]
        half_w = (label[3] * w) / 2
        half_h = (label[4] * h) / 2
        cx = label[1] * w
        cy = label[2] * h
        label[1] = "{:.0f}".format(cx - half_w)
        label[2] = "{:.0f}".format(cy - half_h)
        label[3] = "{:.0f}".format(cx + half_w)
        label[4] = "{:.0f}".format(cy + half_h)
    
    return converted

def convert_yolo_labels(images_dir, labels_dir, target_dir):
    label_list = [label_file for label_file in os.listdir(labels_dir) if isExtension(label_file, ".txt")]

    for label_file in label_list:
        image_name = label_file.replace(".txt", ".jpg")
        image = load_image(os.path.join(images_dir, image_name))
        labels = load_labels(os.path.join(labels_dir, label_file))
        converted = convert_yolo_format(image, labels)
        with open(os.path.join(target_dir, label_file), "w") as fout:
            for converted_label in converted:
                fout.write(" ".join(converted_label) + "\n")
                
def convert_box_to_original(image, bboxs, model_dim):
    img_size = image.size
    ratio = min(model_dim / img_size[0], model_dim / img_size[1])
    new_w = int(img_size[0] * ratio)
    new_h = int(img_size[1] * ratio)
    
    converted = []
    for bbox in bboxs:
        w = (bbox[2] - bbox[0]) * model_dim / ratio
        h = (bbox[3] - bbox[1]) * model_dim / ratio
        left = ((bbox[0] * model_dim - (model_dim - new_w) // 2)) / ratio
        top = (bbox[1] * model_dim - (model_dim - new_h) // 2) / ratio
        right = left + w
        bottom = top + h
        converted.append([left, top, right, bottom])
        
    return converted

def data_generator(images_dir, batch_size=32, vgg_input=False):
    images = [image for image in os.listdir(images_dir) if isExtension(image, ".jpg")]
    
    m = len(images)
    for offset in range(0, m, batch_size):
        X = []
        files = []
        for i in range(offset, min(offset+batch_size, m)):
            file_name = images[i].replace(".jpg", "")
            image = np.array(load_image(os.path.join(images_dir, images[i])))

            if vgg_input:
                image = covert_to_VGG_input(image)
            
            X.append(image)
            files.append(file_name)
            
        X = np.stack(X, axis=0)
        
        yield X, files
        
def write_prediction_to_file(file_path, image_path, scores, boxes, classes):
    image = load_image(image_path)
    boxes = convert_box_to_original(image, boxes, MODEL_DIM)
    
    with open(file_path, "w") as fout:
        for score, box, class_idx in zip(scores, boxes, classes):
            fout.write("{} {:.6f} {:.0f} {:.0f} {:.0f} {:.0f}\n".format(
                CLASS_NAME[class_idx], score, box[0], box[1], box[2], box[3]))
