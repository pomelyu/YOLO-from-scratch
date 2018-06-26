import os
import numpy as np

from .constants import GRID_SIZE, NUM_BOX
from .utils import load_image, isExtension

def batch_generator(imgs_path, labels_path, batch_size=32, random_seed=0):
    imgs = sorted([file for file in os.listdir(imgs_path) if isExtension(file, ".jpg")])
    labels = sorted([label for label in os.listdir(labels_path) if isExtension(label, ".npy")])
    assert len(imgs) == len(labels)
    
    m = len(imgs)
    np.random.seed(random_seed)
    indexs = np.random.permutation(m)
    for offset in range(0, m, batch_size):
        X = []
        Y = []
        batch_index = indexs[offset:min(offset+batch_size, m)]
        for i in batch_index:
            assert imgs[i][:imgs[i].rindex(".")] == labels[i][:labels[i].rindex(".")]
            
            image = load_image(os.path.join(imgs_path, imgs[i]))
            X.append(np.asarray(image))
            
            bboxs = np.load(os.path.join(labels_path, labels[i]))
            bboxs = bboxs.reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))
            Y.append(bboxs)
            
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        
        yield X, Y
