from keras import layers as L
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input

from .constants import MODEL_DIM, GRID_SIZE, NUM_BOX, NUM_CLASS

def yolo_vgg3_model(regularizer=None):
    CELL_DIM = NUM_BOX * (5 + NUM_CLASS)
    initializer = "glorot_normal"
    
    # Input Layer
    X_input = L.Input((MODEL_DIM, MODEL_DIM, 3))
    X = X_input
    
    # 448 x 448 x 3    
    vgg_model = VGG16(include_top=False, weights='imagenet', input_tensor=X)
    for vgg_layer in vgg_model.layers:
        vgg_layer.trainable = False
    
    X = vgg_model.output
    X = L.BatchNormalization(axis=3)(X)
    
    # 7 x 7 x 512
    X = L.Conv2D(512, kernel_size=(1, 1), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer)(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer)(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.Conv2D(1024, kernel_size=(1, 1), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer)(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.Conv2D(1024, kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer)(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    # 7 x 7 x 1024
    X = L.Conv2D(CELL_DIM, kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer)(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    
    X = L.Conv2D(CELL_DIM // 2, kernel_size=(1, 1), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer)(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    
    X = L.Conv2D(CELL_DIM, kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer)(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    
    # 7 x 7 x 100
    X_BBox = L.Conv2D(NUM_BOX * 5, kernel_size=(1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer)(X)
    X_BBox = L.Reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))(X_BBox)
    X_BBox = L.Activation('sigmoid', name="ActBBox")(X_BBox)
    
    X_Class = L.Conv2D(NUM_BOX * NUM_CLASS, kernel_size=(1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer)(X)
    X_Class = L.Reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))(X_Class)
    X_Class = L.Activation('softmax', name="ActClass")(X_Class)
    
    X = L.Concatenate(axis=-1)([X_BBox, X_Class])
    
    model = Model(inputs=X_input, outputs=X, name="yolo_vgg3")

    return model
