import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TerminateOnNaN

from .constants import GRID_SIZE, NUM_BOX, GAMMA_MIN, GAMMA_MAX
from .io import load_image
from .image_transform import flip_image_horizontal, adjust_gamma, covert_to_VGG_input

        
def generator_from_array(labels, images, batch_size=32, vgg_input=False, normalized=False):
    assert len(labels) == len(images)
    m = len(labels)
    
    while True:
        index = np.random.permutation(m)
        for offset in range(0, m, batch_size):
            X = []
            Y = []
            for i in index[offset:offset+batch_size]:
                image = load_image(images[i])
                bboxs = np.load(labels[i])
                    
                # convert image to vgg format
                if vgg_input:
                    image = covert_to_VGG_input(image)
                if normalized:
                    image = image / 255

                X.append(np.asarray(image))
                Y.append(bboxs)
                
            X = np.stack(X, axis=0)
            Y = np.stack(Y, axis=0)
            
            yield X, Y

def train_valid_yolo(model, train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir,
    model_name=None, model_dir="./", batch_size=32, epochs=10, epoch_begin=0, 
    vgg_input=False, normalized=False):

    apply_labels_dir = np.vectorize(lambda label, labels_dir: os.path.join(labels_dir, label))
    apply_images_dir = np.vectorize(lambda label, images_dir: os.path.join(images_dir, label.replace(".npy", ".jpg")))
    
    # Training Generator
    train_labels = np.array([label for label in os.listdir(train_labels_dir) if label.endswith(".npy")])
    train_images = apply_images_dir(train_labels, train_images_dir)
    train_labels = apply_labels_dir(train_labels, train_labels_dir)
    m_train = len(train_labels)
    
    train_generator = generator_from_array(
        train_labels, 
        train_images,
        batch_size=batch_size, 
        vgg_input=vgg_input,
        normalized=normalized)

    # Validation Generator
    valid_labels = np.array([label for label in os.listdir(valid_labels_dir) if label.endswith(".npy")])
    valid_images = apply_images_dir(valid_labels, valid_images_dir)
    valid_labels = apply_labels_dir(valid_labels, valid_labels_dir)
    m_valid = len(valid_labels)

    valid_generator = generator_from_array(
        valid_labels,
        valid_images,
        batch_size=batch_size,
        vgg_input=vgg_input,
        normalized=normalized)

    callbacks = [
        ModelCheckpoint(os.path.join(model_dir, "{model_name}").format(model_name=model_name) + "-{epoch:02d}-{val_loss:.2f}.h5", save_best_only=True),
        TerminateOnNaN(),
    ]
    
    # Training
    model_name = "{}-{}".format(model_name, epoch_begin)
    print("Epoch:", epoch_begin)
    model.fit_generator(
        train_generator,
        epochs=epochs,
        steps_per_epoch=(m_train // batch_size),
        validation_data=valid_generator,
        validation_steps=(m_valid // batch_size),
        initial_epoch=epoch_begin,
        callbacks=callbacks
    )

        

def evaluate_yolo(model, images_dir, labels_dir, batch_size=32, vgg_input=False, normalized=False):
    test_labels = np.array([label for label in os.listdir(labels_dir) if label.endswith(".npy")])
    m = len(test_labels)
    
    apply_labels_dir = np.vectorize(lambda label, labels_dir: os.path.join(labels_dir, label))
    apply_images_dir = np.vectorize(lambda label, images_dir: os.path.join(images_dir, label.replace(".npy", ".jpg")))
    
    test_images = apply_images_dir(test_labels, images_dir)
    test_labels = apply_labels_dir(test_labels, labels_dir)
    
    test_generator = generator_from_array(
        test_labels,
        test_images,
        batch_size=batch_size,
        vgg_input=vgg_input,
        normalized=normalized)
    
    loss = model.evaluate_generator(test_generator, steps=(m // batch_size), verbose=1)
    print("Evaluation loss:", loss)
