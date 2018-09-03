import os
import numpy as np
from skimage import io
from .constants import GRID_SIZE

def load_anchors(path=None):
    # COCO
#     return np.array([
#         [0.57273, 0.67739],
#         [1.87446, 2.06253],
#         [3.33843, 5.47434],
#         [7.88282, 3.52778],
#         [9.77052, 9.16828],
#     ]) / GRID_SIZE
    
    # VOC
    return np.array([
        [1.32210, 1.73145],
        [3.19275, 4.00944],
        [5.05587, 8.09892],
        [9.47112, 4.84053],
        [11.2364, 10.0071],
    ])

def load_image(path):
    return io.imread(path)

def load_label(path):
    labels = []
    with open(path, 'r') as fin:
        for line in fin:
            obj = [ float(num) for num in line.split(" ") ]
            labels.append(obj)

    return np.array(labels)

def load_class_label(path):
    labels = []
    with open(path, 'r') as fin:
        for line in fin:
            obj = [ num for num in line.split(" ") ]
            labels.append(obj)

    return labels

def save_image(path, image):
    io.imsave(path, image)

def save_npy(path, npy_array):
    np.save(path, npy_array)

def save_label(labels, path):
    with open(path, "w") as fout:
        for label in labels:
            fout.write("{:.0f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                label[1], label[2], label[3], label[4], label[5]))

def save_score_label(labels, path):
    with open(path, "w") as fout:
        for label in labels:
            fout.write("{:.4f} {:.0f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                label[0], label[1], label[2], label[3], label[4], label[5]))

def save_class_label(labels, image_shape, classes, path, with_score=True):
    h, w = image_shape
    with open(path, "w") as fout:
        for label in labels:
            box = label[2:] if with_score else label[1:]
            cx = box[0] * w
            cy = box[1] * h
            w2 = (box[2] * w) // 2
            h2 = (box[3] * h) // 2
            if with_score:
                fout.write("{} {:.4f} {:.0f} {:.0f} {:.0f} {:.0f}\n".format(
                    classes[int(label[1])], label[0], cx-w2, cy-h2, cx+w2, cy+h2))
            else:
                fout.write("{} {:.0f} {:.0f} {:.0f} {:.0f}\n".format(
                    classes[int(label[0])], cx-w2, cy-h2, cx+w2, cy+h2))

def image_label_generator(image_dir, lable_dir):
    all_files = [label.replace(".txt", "") for label in os.listdir(lable_dir) if label.endswith(".txt")]
    
    for file_name in all_files:
        image = load_image(os.path.join(image_dir, "{}.jpg".format(file_name)))
        label = load_label(os.path.join(lable_dir, "{}.txt".format(file_name)))

        yield file_name, image, label

