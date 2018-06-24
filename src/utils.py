import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from .constants import OBJECT_CLASS

def draw_label_rect(name, cx, cy, w, h, ax, color="blue"):
    x = cx - (w / 2)
    y = cy - (h / 2)
    ax.text(x, y, name, fontsize=12, horizontalalignment="left", 
        verticalalignment="top", backgroundcolor=color, color="white")
    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor=color))


def visualize_label(img_path, label_path):
    """ display the bounding box and classes of objects in the image

    Arguments:
    img_path: relative path of the image
    label_path: relative path of the text for bounding box information
        line example: classname x y w h (x, y, w, h is normalized)

    Example:
    visualize_label("../data/VOC2007/images/000007.jpg", "../data/VOC2007/labels/000007.txt")

    """

    cwd = os.getcwd()
    img = plt.imread(os.path.join(cwd, img_path))
    
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_title(img_path)
    ax.imshow(img)
    
    h, w, _ = img.shape
    with open(os.path.join(cwd, label_path), 'r') as fin:
        for line in fin:
            obj = [ float(num) for num in line.split(" ") ]
            class_name = OBJECT_CLASS[int(obj[0])]
            draw_label_rect(class_name, obj[1] * w, obj[2] * h, obj[3] * w, obj[4] * h, ax)

    plt.show()
