import tensorflow as tf
import keras.backend as K
import numpy as np

from .constants import GRID_SIZE, NUM_BOX, NUM_CLASS

def split_bbox(Y):
    """ Split the predicted result to three tensor: box_confidence, boxes, box_class_probs
    
    Arguments:
        Y: [GRID_SIZE, GRID_SIZE, NUM_BOX, 5 + NUM_CLASS] tensor - output of yolo neural network
    
    Returns:
        box_confidence: [m , GRID_SIZE, GRID_SIZE, NUM_BOX, 1] tensor
        boxes: [m, GRID_SIZE, GRID_SIZE, NUM_BOX, 4] tensor - bbox dimension in cell corrdinates
        box_class_probs: [m, GRID_SIZE, GRID_SIZE, NUM_BOX, NUM_CLASS] tensor
    """
    box_confidence = Y[:, :, :, 0:1]
    boxes = Y[:, :, :, 1:5]
    box_class_probs = Y[:, :, :, 5:]
    
    return box_confidence, boxes, box_class_probs

def bbox_cell_to_global(boxes):
    """ Covert bbox from cell coordinates(centeor) to image coordinates(corner) to perform non_maximum_supress
    
    Argumens:
        boxes: [GRID_SIZE, GRID_SIZE, NUM_BOX, 4] - center_x_cell, center_y_cell, image_w, image_h
        
    Returns:
        boxes: [GRID_SIZE, GRID_SIZE, NUM_BOX, 4] - corner_x1, corner_y1, corner_x2, corner_y2
    """
    cell_offset_x = tf.reshape(tf.constant(
        np.tile(np.array(range(GRID_SIZE)), (GRID_SIZE, 1)),
        dtype="float32"), [GRID_SIZE, GRID_SIZE, 1, 1])
    
    cell_offset_y = tf.reshape(tf.constant(
        np.tile(np.array(range(GRID_SIZE)).reshape((-1, 1)), (1, GRID_SIZE)),
        dtype="float32"), [GRID_SIZE, GRID_SIZE, 1, 1])
    
    # why not
    # center_x = (boxes[:, :, :, 0:1] + cell_offset_y) / GRID_SIZE
    # center_y = (boxes[:, :, :, 1:2] + cell_offset_x) / GRID_SIZE
    center_x = (boxes[:, :, :, 0:1] + cell_offset_y) / GRID_SIZE
    center_y = (boxes[:, :, :, 1:2] + cell_offset_x) / GRID_SIZE
    half_w = boxes[:, :, :, 2:3] / 2
    half_h = boxes[:, :, :, 3:4] / 2
    
    corner_x1 = center_x - half_w
    corner_y1 = center_y - half_h
    corner_x2 = center_x + half_w
    corner_y2 = center_y + half_h
    
    return tf.concat([corner_x1, corner_y1, corner_x2, corner_y2], axis=-1)


def filter_bbox_by_scores(box_confidence, boxes, box_class_probs, threshold=0.6):
    """ Filter out the bounding box with confidence lower than threshold and return the prediced class
    
    Arguments:
        box_confidence: [GRID_SIZE, GRID_SIZE, NUM_BOX, 1] tensor
        boxes: [GRID_SIZE, GRID_SIZE, NUM_BOX, 4] tensor - bbox dimension in cell corrdinates
        box_class_probs: [GRID_SIZE, GRID_SIZE, NUM_BOX, NUM_CLASS] tensor
        threshold: the confidence threshold
        
    Returns:
        scores: [None, 1]
        boxes: [None, 4]
        classes: [None, 1]
    """
    box_scores = tf.multiply(box_confidence, box_class_probs)
    
    box_classes = tf.argmax(box_scores, axis=-1)
    box_class_scores = tf.reduce_max(box_scores, axis=-1)
    
    filtering_mask = box_class_scores >= threshold
    
    
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes

def non_maximum_supress(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """ Perform non maximum suppress to remove the overlapped boxes
    
    Arguments:
        scores: [None, 1]
        boxes: [None, 4]
        classes: [None, 1]
        max_boxes: maximum boxes to return
        iou_threshold: the threshold to evaluated overlapped boxes
    
    Returns:
        scores: [None, 1]
        boxes: [None, 4]
        classes: [None, 1]
    """
    
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes


def postprocessing(sess, Y_pred, conf_threshold=0.6, iou_threshold=0.5):
    """ Perform thresholding and non-maximum supression to the neural network output

    Argumens:
        sess: tensorflow Session
        Y_pred: [GRID_SIZE, GRID_SIZE, NUM_BOX, 5 + NUM_CLASS] tensor - output of neural network
        conf_threshold: the confidence threshold
        iou_threshold: the threshold to evaluated overlapped boxes


    """
    
    Y = tf.placeholder("float32", shape=(GRID_SIZE, GRID_SIZE, NUM_BOX, 5 + NUM_CLASS))
    
    box_confidence, boxes, box_class_probs = split_bbox(Y)
    boxes = bbox_cell_to_global(boxes)
    
    # chekck dimension
    tf.assert_equal(box_confidence.shape.as_list(), [GRID_SIZE, GRID_SIZE, NUM_BOX, 1])
    tf.assert_equal(boxes.shape.as_list(), [GRID_SIZE, GRID_SIZE, NUM_BOX, 4])
    tf.assert_equal(box_class_probs.shape.as_list(), [GRID_SIZE, GRID_SIZE, NUM_BOX, NUM_CLASS])
    
    scores, boxes, classes = filter_bbox_by_scores(
        box_confidence, boxes, box_class_probs, threshold=conf_threshold)
    scores, boxes, classes = non_maximum_supress(
        scores, boxes, classes, iou_threshold=iou_threshold)
    
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={ Y: Y_pred })
    
    return out_scores, out_boxes, out_classes
