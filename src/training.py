import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TerminateOnNaN

from .constants import GRID_SIZE, NUM_BOX, GAMMA_MIN, GAMMA_MAX, MODEL_DIM
from .io import load_image, load_label
from .preprocess import ResizeTransform, bbox_2_bbox_map
from .image_transform import covert_to_VGG_input

def data_generator(image_dir, label_dir, anchors, batch_size=32, shuffle=True, augment=False, 
    vgg_input=False, normalized=False):

    files = np.array([label.replace(".txt", "") for label in os.listdir(label_dir) if label.endswith(".txt")])
    n = len(files)

    resize_transform = ResizeTransform()
    while True:
        if shuffle:
            files = files[np.random.permutation(n)]

        for offset in range(0, n, batch_size):
            images = []
            labels = []
            for file_name in files[offset:offset+batch_size]:
                image = load_image(os.path.join(image_dir, "{}.jpg".format(file_name)))
                label = load_label(os.path.join(label_dir, "{}.txt".format(file_name)))

                image, label[:, 1:] = resize_transform.transform(image, label[:, 1:])
                label = bbox_2_bbox_map(label, anchors)

                if vgg_input:
                    image = covert_to_VGG_input(image)
                if normalized:
                    image = image / 255

                images.append(image)
                labels.append(label)

            yield np.stack(images, axis=0), np.stack(labels, axis=0)


def train_valid_yolo(model, anchors, train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir,
    model_name=None, model_dir="./", batch_size=32, epochs=10, epoch_begin=0, 
    augment=False, vgg_input=False, normalized=False):
    
    # Training Generator
    m_train = len([label for label in os.listdir(train_labels_dir) if label.endswith(".txt")])
    
    train_generator = data_generator(
        train_images_dir, 
        train_labels_dir,
        anchors=anchors,
        shuffle=True,
        augment=augment,
        batch_size=batch_size, 
        vgg_input=vgg_input,
        normalized=normalized)

    # Validation Generator
    m_valid = len([label for label in os.listdir(valid_labels_dir) if label.endswith(".txt")])

    valid_generator = data_generator(
        valid_images_dir,
        valid_labels_dir,
        anchors=anchors,
        shuffle=False,
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

        

def evaluate_yolo(model, anchors, images_dir, labels_dir, batch_size=32, vgg_input=False, normalized=False):
    test_labels = np.array([label for label in os.listdir(labels_dir) if label.endswith(".txt")])
    m = len(test_labels)
    
    test_generator = data_generator(
        images_dir,
        labels_dir,
        anchors=anchors,
        shuffle=False,
        batch_size=batch_size,
        vgg_input=vgg_input,
        normalized=normalized)
    
    loss = model.evaluate_generator(test_generator, steps=(m // batch_size), verbose=1)
    print("Evaluation loss:", loss)
