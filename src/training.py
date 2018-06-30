import os
import numpy as np

from .constants import GRID_SIZE, NUM_BOX
from .utils import load_image, isExtension

def flip_data_horizontal(image, bbox):
    flipped_image = np.flip(image, axis=1)
    flipped_bbox = np.flip(bbox, axis=0)
    flipped_bbox[:, :, :, 1] = 1 - flipped_bbox[:, :, :, 1]
    
    return flipped_image, flipped_bbox

def batch_generator(imgs_path, labels_path, batch_size=32, random_seed=0, argument_data=True):
    labels = [label for label in os.listdir(labels_path) if isExtension(label, ".npy")]
    
    m = len(labels)
    np.random.seed(random_seed)
    indexs = np.random.permutation(m)
    for offset in range(0, m, batch_size):
        X = []
        Y = []
        batch_index = indexs[offset:min(offset+batch_size, m)]
        for i in batch_index:
            image_name = labels[i].replace(".npy", ".jpg")
            image = load_image(os.path.join(imgs_path, image_name))

            bboxs = np.load(os.path.join(labels_path, labels[i]))
            bboxs = bboxs.reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))
            
            # Data argumentation: Flip
            np.random.seed(random_seed + i)
            to_flip = (np.random.rand() < 0.5)
            if argument_data and to_flip:
                image, bboxs = flip_data_horizontal(image, bboxs)

            X.append(np.asarray(image))
            Y.append(bboxs)
            
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        
        yield X, Y

def train_yolo(model, train_X, train_Y, epochs=10, batch_size=32, epoch_begin=0):
    m = len([file for file in os.listdir(train_X) if isExtension(file, ".jpg")])

    for i in range(epochs):
        print("Epoch:", epoch_begin + i)
        data_stream = batch_generator(
            train_X,
            train_Y,
            batch_size=batch_size,
            random_seed=epoch_begin + i)
    
        model.fit_generator(data_stream, steps_per_epoch=(m // batch_size))


def evaluate_yolo(model, X, Y, batch_size=32):
    m = len([file for file in os.listdir(X) if isExtension(file, ".jpg")])

    data_stream = batch_generator(
        X,
        Y,
        batch_size=batch_size,
        argument_data=False)
    loss = model.evaluate_generator(data_stream, steps=(m // batch_size), verbose=1)
    print("Evaluation loss:", loss)
