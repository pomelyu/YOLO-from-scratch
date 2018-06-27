from keras import layers as L
from keras.models import Model

from .loss import create_yolo1_loss
from .constants import MODEL_DIM, GRID_SIZE, NUM_BOX, NUM_CLASS

def yolo_tiny_model(optimizer="adam", lambda_coord=5, lambda_noobj=0.5):
    # Input Layer
    X_input = L.Input((MODEL_DIM, MODEL_DIM, 3))
    X = X_input
    
    # 448 x 448 x 3
    X = L.Conv2D(16, kernel_size=(3, 3), padding="same", name="Conv1")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    # 224 x 224 x 16
    X = L.Conv2D(32, kernel_size=(3, 3), padding="same", name="Conv2")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    # 112 x 112 x 32
    X = L.Conv2D(64, kernel_size=(3, 3), padding="same", name="Conv3")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    # 56 x 56 x 64
    X = L.Conv2D(128, kernel_size=(3, 3), padding="same", name="Conv4")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    # 28 x 28 x 128
    X = L.Conv2D(256, kernel_size=(3, 3), padding="same", name="Conv5")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    # 14 x 14 x 256
    X = L.Conv2D(512, kernel_size=(3, 3), padding="same", name="Conv6")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    # 7 x 7 x 512
    X = L.Conv2D(NUM_BOX * (5 + NUM_CLASS), kernel_size=(1, 1), padding="same", name="Conv7")(X)
    X = L.BatchNormalization(axis=3)(X)
    X = L.LeakyReLU()(X)
    
    # 7 x 7 x 100
    X_BBox = L.Conv2D(NUM_BOX * 5, kernel_size=(1, 1), name="ConvBBox")(X)
    X_BBox = L.Reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))(X_BBox)
    X_BBox = L.Activation('sigmoid', name="ActBBox")(X_BBox)
    
    X_Class = L.Conv2D(NUM_BOX * NUM_CLASS, kernel_size=(1, 1), name="ConvClass")(X)
    X_Class = L.Reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, -1))(X_Class)
    X_Class = L.Activation('softmax', name="ActClass")(X_Class)
    
    X = L.Concatenate(axis=-1)([X_BBox, X_Class])
    
    model = Model(inputs=X_input, outputs=X, name="Yolo_tiny")
    
    # Loss function
    yolo1_loss = create_yolo1_loss(lambda_coord, lambda_noobj)

    model.compile(optimizer="adam", loss=yolo1_loss)

    return model
