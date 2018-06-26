import os
import numpy as np

from .constants import GRID_SIZE, NUM_BOX
from .utils import load_image, isExtension

def batch_generator(imgs_path, labels_path, batch_size=32):
    imgs = sorted([file for file in os.listdir(imgs_path) if isExtension(file, ".jpg")])
    labels = sorted([label for label in os.listdir(labels_path) if isExtension(label, ".npy")])
    assert len(imgs) == len(labels)
    
    m = len(imgs)
    for i in range(0, m, batch_size):
        X = []
        Y = []
        for j in range(i, min(i+batch_size, m)):
            assert imgs[j][:imgs[j].rindex(".")] == labels[j][:labels[j].rindex(".")]
            
            image = load_image(os.path.join(imgs_path, imgs[j]))
            X.append(np.asarray(image))
            
            bboxs = np.load(os.path.join(labels_path, labels[j]))
            bboxs = bboxs.reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))
            Y.append(bboxs)
            
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        
        yield X, Y
