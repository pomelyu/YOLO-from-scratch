import tensorflow as tf
from keras import layers as L
from keras.models import Model

from src.constants import MODEL_DIM, GRID_SIZE, NUM_BOX, NUM_CLASS

def yolo2_model():
    
    # Input Layer
    X_input = L.Input((MODEL_DIM, MODEL_DIM, 3))
    X = X_input
    
    # 448 x 448 x 3
    X = L.Conv2D(32, kernel_size=(3, 3), padding="same", use_bias=False, trainable=False, name="conv_1")(X)
    X = L.BatchNormalization(name="norm_1", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    
    X = L.Conv2D(64, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_2")(X)
    X = L.BatchNormalization(name="norm_2", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    
    X = L.Conv2D(128, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_3")(X)
    X = L.BatchNormalization(name="norm_3", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(64, (1, 1), padding="same", use_bias=False, trainable=False , name="conv_4")(X)
    X = L.BatchNormalization(name="norm_4", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(128, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_5")(X)
    X = L.BatchNormalization(name="norm_5", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    
    X = L.Conv2D(256, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_6")(X)
    X = L.BatchNormalization(name="norm_6", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(128, (1, 1), padding="same", use_bias=False, trainable=False , name="conv_7")(X)
    X = L.BatchNormalization(name="norm_7", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(256, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_8")(X)
    X = L.BatchNormalization(name="norm_8", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    
    X = L.Conv2D(512, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_9")(X)
    X = L.BatchNormalization(name="norm_9", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(256, (1, 1), padding="same", use_bias=False, trainable=False , name="conv_10")(X)
    X = L.BatchNormalization(name="norm_10", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(512, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_11")(X)
    X = L.BatchNormalization(name="norm_11", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(256, (1, 1), padding="same", use_bias=False, trainable=False , name="conv_12")(X)
    X = L.BatchNormalization(name="norm_12", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(512, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_13")(X)
    X = L.BatchNormalization(name="norm_13", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X_Residual = X
    X = L.MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    X = L.Conv2D(1024, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_14")(X)
    X = L.BatchNormalization(name="norm_14", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(512, (1, 1), padding="same", use_bias=False, trainable=False , name="conv_15")(X)
    X = L.BatchNormalization(name="norm_15", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(1024, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_16")(X)
    X = L.BatchNormalization(name="norm_16", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(512, (1, 1), padding="same", use_bias=False, trainable=False , name="conv_17")(X)
    X = L.BatchNormalization(name="norm_17", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(1024, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_18")(X)
    X = L.BatchNormalization(name="norm_18", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(1024, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_19")(X)
    X = L.BatchNormalization(name="norm_19", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(1024, (3, 3), padding="same", use_bias=False, trainable=False , name="conv_20")(X)
    X = L.BatchNormalization(name="norm_20", trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    # Residual Connection
    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)
    
    X_Residual = L.Conv2D(64, (1, 1), padding="same", use_bias=False, trainable=False, name="conv_21")(X_Residual)
    X_Residual = L.BatchNormalization(name="norm_21", trainable=False)(X_Residual)
    X_Residual = L.LeakyReLU(alpha=0.1)(X_Residual)
    X_Residual = L.Lambda(space_to_depth_x2)(X_Residual)
    
    X = L.Concatenate(axis=-1)([X_Residual, X])
    
    X = L.Conv2D(1024, (3, 3), padding="same", use_bias=False, trainable=False, name="conv_22")(X)
    X = L.BatchNormalization(name='norm_22', trainable=False)(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Conv2D(NUM_BOX * (NUM_CLASS + 5), (3, 3), padding="same", use_bias=False, name="conv_top")(X)
    X = L.BatchNormalization(name='norm_top')(X)
    X = L.LeakyReLU(alpha=0.1)(X)
    
    X = L.Reshape((GRID_SIZE, GRID_SIZE, NUM_BOX, NUM_CLASS + 5))(X)
    
    model = Model(inputs=X_input, outputs=X, name="yolo2")

    return model
