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

def train_yolo(model, train_X_list, train_Y_list, epochs=10, batch_size=32, epoch_begin=0):
    for i in range(epochs):
        print("Epoch:", epoch_begin + i)
        np.random.seed(epoch_begin + i)
        dataset_index = np.random.permutation(len(train_X_list))
        for i in dataset_index:
            train_X = train_X_list[i]
            train_Y = train_Y_list[i]
            m = len([file for file in os.listdir(train_X) if isExtension(file, ".jpg")])
            data_stream = batch_generator(
                train_X,
                train_Y,
                batch_size=batch_size,
                random_seed=epoch_begin + i)

            model.fit_generator(data_stream, steps_per_epoch=(m // batch_size))

def train_valid_yolo(model, images_dir, labels_dir, valid_ratio=0.2, batch_size=32, epochs=10, epoch_begin=0):
    for i in range(epochs):
        random_seed = epoch_begin + i
        print("Epoch:", epoch_begin+i)
        np.random.seed(random_seed)
        
        labels_file = np.array([label for label in os.listdir(labels_dir) if isExtension(label, ".npy")])
        
        # shuffle and split train/valid
        m = len(labels_file)
        labels_file = labels_file[np.random.permutation(m)]
        train_labels = labels_file[:-int(m*valid_ratio)]
        valid_labels = labels_file[-int(m*valid_ratio):]
        
        apply_labels_dir = np.vectorize(lambda label: os.path.join(labels_dir, label))
        apply_images_dir = np.vectorize(lambda label: os.path.join(images_dir, label.replace(".npy", ".jpg")))
        
        train_images = apply_images_dir(train_labels)
        train_labels = apply_labels_dir(train_labels)
        valid_images = apply_images_dir(valid_labels)
        valid_labels = apply_labels_dir(valid_labels)
        
        train_generator = generator_from_array(
            train_labels, 
            train_images, 
            batch_size=batch_size, 
            random_seed=random_seed)
        
        valid_generator = generator_from_array(
            valid_labels,
            valid_images,
            batch_size=batch_size,
            random_seed=random_seed)
        
        model.fit_generator(
            train_generator,
            steps_per_epoch=(len(train_images) // batch_size),
            validation_data=valid_generator,
            validation_steps=(len(valid_images) // batch_size))
        
        

def evaluate_yolo(model, X, Y, batch_size=32):
    m = len([file for file in os.listdir(X) if isExtension(file, ".jpg")])

    data_stream = batch_generator(
        X,
        Y,
        batch_size=batch_size,
        argument_data=False)
    loss = model.evaluate_generator(data_stream, steps=(m // batch_size), verbose=1)
    print("Evaluation loss:", loss)
