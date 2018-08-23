import os
import numpy as np
from skimage import io

def load_anchors(path=None):
    return np.array([
        [0.57273, 0.67739],
        [1.87446, 2.06253],
        [3.33843, 5.47434],
        [7.88282, 3.52778],
        [9.77052, 9.16828],
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

def image_label_generator(image_dir, lable_dir):
    all_files = [label.replace(".txt", "") for label in os.listdir(lable_dir) if label.endswith(".txt")]
    
    for file_name in all_files:
        image = load_image(os.path.join(image_dir, "{}.jpg".format(file_name)))
        label = load_label(os.path.join(lable_dir, "{}.txt".format(file_name)))

        yield file_name, image, label

