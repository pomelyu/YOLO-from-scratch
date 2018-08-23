import tensorflow as tf
import numpy as np
from .constants import GRID_SIZE

def create_yolo_loss(anchors, lambda_coord=5, lambda_noobj=0.5):
    def yolo_loss(y_true, y_pred):
        lc = tf.constant(lambda_coord, dtype="float32")
        ln = tf.constant(lambda_noobj, dtype="float32")
        
        grids = np.arange(GRID_SIZE)
        cell_x = tf.constant(
            np.tile(grids.reshape((-1, 1)), (1, GRID_SIZE)).reshape((GRID_SIZE, GRID_SIZE, 1)),
            dtype="float32"
        )
        cell_y = tf.constant(
            np.tile(grids, (GRID_SIZE, 1)).reshape((GRID_SIZE, GRID_SIZE, 1)),
            dtype="float32"
        )
        anchor_w = tf.constant(
            np.tile(anchors[:, 0], (GRID_SIZE, GRID_SIZE, 1)),
            dtype="float32"
        )
        anchor_h = tf.constant(
            np.tile(anchors[:, 1], (GRID_SIZE, GRID_SIZE, 1)),
            dtype="float32"
        )

        pred_c = tf.sigmoid(y_pred[..., 0])
        pred_x = (tf.sigmoid(y_pred[..., 1]) + cell_x) / GRID_SIZE
        pred_y = (tf.sigmoid(y_pred[..., 2]) + cell_y) / GRID_SIZE
        pred_w = anchor_w * tf.exp(y_pred[..., 3])
        pred_h = anchor_h * tf.exp(y_pred[..., 4])
        pred_class = tf.nn.softmax(y_pred[..., 5:], axis=-1)

        obj_mask = y_true[..., 0]
        loss_c = tf.square(y_true[..., 0] - pred_c)
        loss_x = tf.square(y_true[..., 1] - pred_x)
        loss_y = tf.square(y_true[..., 2] - pred_y)
        loss_w = tf.square(tf.sqrt(y_true[..., 3]) - tf.sqrt(pred_w))
        loss_h = tf.square(tf.sqrt(y_true[..., 4]) - tf.sqrt(pred_h))
        loss_class = tf.reduce_sum(tf.square(y_true[..., 5:] - pred_class), axis=-1)
        
        loss_bbox = tf.reduce_sum(lc * obj_mask * (loss_x + loss_y + loss_w + loss_h))
        loss_conf = tf.reduce_sum(obj_mask * loss_c) + tf.reduce_sum(ln * (1 - obj_mask) * loss_c)
        loss_class = tf.reduce_sum(obj_mask * loss_class)
        
        return (loss_bbox + loss_conf + loss_class) / tf.cast(tf.shape(y_true)[0], 'float32')

    return yolo_loss
