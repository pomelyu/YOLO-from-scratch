import tensorflow as tf

def create_yolo1_loss(lambda_coord=5, lambda_noobj=0.5):
    def yolo1_loss(y_true, y_pred):
        lc = tf.constant(lambda_coord, dtype="float32")
        ln = tf.constant(lambda_noobj, dtype="float32")
        
        obj_mask = y_true[:, :, :, :, 0]
        loss_c = tf.square(y_true[:, :, :, :, 0] - y_pred[:, :, :, :, 0])
        loss_x = tf.square(y_true[:, :, :, :, 1] - y_pred[:, :, :, :, 1])
        loss_y = tf.square(y_true[:, :, :, :, 2] - y_pred[:, :, :, :, 2])
        loss_w = tf.square(tf.sqrt(y_true[:, :, :, :, 3]) - tf.sqrt(y_pred[:, :, :, :, 3]))
        loss_h = tf.square(tf.sqrt(y_true[:, :, :, :, 4]) - tf.sqrt(y_pred[:, :, :, :, 4]))
        loss_class = tf.reduce_sum(tf.square(y_true[:, :, :, :, 5:] - y_pred[:, :, :, :, 5:]), axis=-1)
        
        loss_bbox = tf.reduce_sum(lc * obj_mask * (loss_x + loss_y + loss_w + loss_h))
        loss_conf = tf.reduce_sum(obj_mask * loss_c) + tf.reduce_sum(ln * (1 - obj_mask) * loss_c)
        loss_class = tf.reduce_sum(obj_mask * loss_class)
        
        return (loss_bbox + loss_conf + loss_class) / tf.cast(tf.shape(y_true)[0], 'float32')

    return yolo1_loss
