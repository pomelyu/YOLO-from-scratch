from keras import layers as L
from keras.models import Model
from keras.applications.vgg16 import VGG16

from .constants import MODEL_DIM, GRID_SIZE, NUM_BOX, NUM_CLASS

def yolo_vgg_model(regularizer=None):
    CELL_DIM = NUM_BOX * (5 + NUM_CLASS)
    
    # Input Layer
    X_input = L.Input((MODEL_DIM, MODEL_DIM, 3))
    X = X_input
    
    # 448 x 448 x 3
    X = L.Conv2D(64, kernel_size=(7, 7), padding="same", strides=(2, 2), name="Conv1")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    vgg_model = VGG16(include_top=False, weights='imagenet')
    for vgg_layer in vgg_model.layers[4:]:
        X = vgg_layer(X)
    
    # 7 x 7 x 512
    X = L.Conv2D(CELL_DIM, kernel_size=(3, 3), padding="same", kernel_regularizer=regularizer, name="Conv7")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    
    X = L.Conv2D(CELL_DIM // 2, kernel_size=(1, 1), padding="same", kernel_regularizer=regularizer, name="Conv8")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    
    X = L.Conv2D(CELL_DIM, kernel_size=(3, 3), padding="same", kernel_regularizer=regularizer, name="Conv9")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    
    # 7 x 7 x 100
    X_BBox = L.Conv2D(NUM_BOX * 5, kernel_size=(1, 1), kernel_regularizer=regularizer, name="ConvBBox")(X)
    X_BBox = L.Reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))(X_BBox)
    X_BBox = L.Activation('sigmoid', name="ActBBox")(X_BBox)
    
    X_Class = L.Conv2D(NUM_BOX * NUM_CLASS, kernel_size=(1, 1), kernel_regularizer=regularizer, name="ConvClass")(X)
    X_Class = L.Reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))(X_Class)
    X_Class = L.Activation('softmax', name="ActClass")(X_Class)
    
    X = L.Concatenate(axis=-1)([X_BBox, X_Class])
    
    model = Model(inputs=X_input, outputs=X, name="yolo_vgg")

    return model
