import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image

from .constants import OBJECT_CLASS

def load_image(image_path):
    """ load image from file
    Arguments:
    image_path: relative path of the image

    Return:
    image: PIL image object
    """
    cwd = os.getcwd()
    img = Image.open(os.path.join(cwd, image_path))

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
    cwd = os.getcwd()

    labels = []
    with open(os.path.join(cwd, label_path), 'r') as fin:
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
        class_name = OBJECT_CLASS[label[0]]
        draw_label_rect(class_name, label[1] * w, label[2] * h, label[3] * w, label[4] * h, ax)

    plt.show()
