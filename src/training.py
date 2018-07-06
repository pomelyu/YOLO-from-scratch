import os
import numpy as np

from .constants import GRID_SIZE, NUM_BOX
from .utils import load_image, isExtension
from .image_transform import flip_image_horizontal, covert_to_VGG_input

        
def generator_from_array(labels, images, batch_size=32, random_seed=0, argument_data=True):
    assert len(labels) == len(images)
    m = len(labels)
    
    for offset in range(0, m, batch_size):
        X = []
        Y = []
        for i in range(offset, min(offset+batch_size, m)):
            image = np.array(load_image(images[i]))
            
            bboxs = np.load(labels[i])
            bboxs = bboxs.reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))
            
            # Data argumentation: Flip
            np.random.seed(random_seed + i)
            to_flip = (np.random.rand() < 0.5)
            if argument_data and to_flip:
                image, bboxs = flip_image_horizontal(image, bboxs)
                
                
            # convert image to vgg format
            image = covert_to_VGG_input(image)

            X.append(np.asarray(image))
            Y.append(bboxs)
            
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        
        yield X, Y

def train_valid_yolo(model, train_images_dirs, train_labels_dirs, valid_images_dir, valid_labels_dir, valid_ratio=0.2, batch_size=32, epochs=10, epoch_begin=0):
    apply_labels_dir = np.vectorize(lambda label, labels_dir: os.path.join(labels_dir, label))
    apply_images_dir = np.vectorize(lambda label, images_dir: os.path.join(images_dir, label.replace(".npy", ".jpg")))
    
    
    assert len(train_labels_dirs) == len(train_images_dirs)
    
    train_labels = np.array([])
    train_images = np.array([])
    for dataset_index in range(len(train_labels_dirs)):
        labels_dir = train_labels_dirs[dataset_index]
        images_dir = train_images_dirs[dataset_index]
        
        train_labels_dataset = np.array([
            label for label in os.listdir(labels_dir) if isExtension(label, ".npy")
        ])
        
        train_images_dataset = apply_images_dir(train_labels_dataset, images_dir)
        train_labels_dataset = apply_labels_dir(train_labels_dataset, labels_dir)
        
        train_labels = np.concatenate((train_labels, train_labels_dataset))
        train_images = np.concatenate((train_images, train_images_dataset))
    
    m_train = len(train_labels)
    
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
        

def evaluate_yolo(model, images_dir, labels_dir, batch_size=32):
    test_labels = np.array([label for label in os.listdir(labels_dir) if isExtension(label, ".npy")])
    m = len(test_labels)
    
    apply_labels_dir = np.vectorize(lambda label, labels_dir: os.path.join(labels_dir, label))
    apply_images_dir = np.vectorize(lambda label, images_dir: os.path.join(images_dir, label.replace(".npy", ".jpg")))
    
    test_images = apply_images_dir(test_labels, images_dir)
    test_labels = apply_labels_dir(test_labels, labels_dir)
    
    test_generator = generator_from_array(
        test_labels,
        test_images,
        batch_size=batch_size,
        argument_data=False)
    
    loss = model.evaluate_generator(test_generator, steps=(m // batch_size), verbose=1)
    print("Evaluation loss:", loss)
