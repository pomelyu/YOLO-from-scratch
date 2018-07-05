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
        
def generator_from_array(labels, images, batch_size=32, random_seed=0, argument_data=True):
    assert len(labels) == len(images)
    
    m = len(labels)
    for offset in range(0, m, batch_size):
        X = []
        Y = []
        for i in range(offset, min(offset+batch_size, m)):
            image = load_image(images[i])

            bboxs = np.load(labels[i])
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

def train_valid_yolo(model, train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir, valid_ratio=0.2, batch_size=32, epochs=10, epoch_begin=0):
    apply_labels_dir = np.vectorize(lambda label, labels_dir: os.path.join(labels_dir, label))
    apply_images_dir = np.vectorize(lambda label, images_dir: os.path.join(images_dir, label.replace(".npy", ".jpg")))
    
    train_labels = np.array([label for label in os.listdir(train_labels_dir) if isExtension(label, ".npy")])
    m_train = len(train_labels)
    train_images = apply_images_dir(train_labels, train_images_dir)
    train_labels = apply_labels_dir(train_labels, train_labels_dir)
    
    valid_labels = np.array([label for label in os.listdir(valid_labels_dir) if isExtension(label, ".npy")])
    m_valid = len(valid_labels)
    valid_images = apply_images_dir(valid_labels, valid_images_dir)
    valid_labels = apply_labels_dir(valid_labels, valid_labels_dir)
    
    np.random.seed(0)
    valid_index = np.random.permutation(m_valid)[:int(m_valid*valid_ratio)]
    
    for i in range(epochs):
        print("Epoch:", epoch_begin+i)
        
        random_seed = epoch_begin + i
        np.random.seed(random_seed)
        
        train_index = np.random.permutation(m_train)
        train_generator = generator_from_array(
            train_labels[train_index], 
            train_images[train_index], 
            batch_size=batch_size, 
            random_seed=random_seed)
        
        valid_generator = generator_from_array(
            valid_labels[valid_index],
            valid_images[valid_index],
            batch_size=batch_size,
            argument_data=False)

        model.fit_generator(
            train_generator,
            steps_per_epoch=(m_train // batch_size),
            validation_data=valid_generator,
            validation_steps=(len(valid_index) // batch_size))
        
        

def evaluate_yolo(model, X, Y, batch_size=32):
    m = len([file for file in os.listdir(X) if isExtension(file, ".jpg")])

    data_stream = batch_generator(
        X,
        Y,
        batch_size=batch_size,
        argument_data=False)
    loss = model.evaluate_generator(data_stream, steps=(m // batch_size), verbose=1)
    print("Evaluation loss:", loss)
