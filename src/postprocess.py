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
    box_confidence = tf.sigmoid(Y[:, :, :, 0:1])
    boxes = Y[:, :, :, 1:5]
    box_class_probs = tf.nn.softmax(Y[:, :, :, 5:], axis=-1)
    
    return box_confidence, boxes, box_class_probs

def bbox_cell_to_global(boxes, anchors):
    """ Covert bbox from cell coordinates(centeor) to image coordinates(corner) to perform non_maximum_supress
    
    Argumens:
        boxes: [GRID_SIZE, GRID_SIZE, NUM_BOX, 4] - center_x_cell, center_y_cell, image_w, image_h
        
    Returns:
        boxes: [GRID_SIZE, GRID_SIZE, NUM_BOX, 4] - corner_x1, corner_y1, corner_x2, corner_y2
    """    
    # In bbox matrix, x coordinate is first axis(axis=0, column)
    # but in the image, x coordinate is horizontal. hence offset_x is 
    # [[1, 1, ... , 1], 
    #   2, 2, ... , 2],
    #   ...
    #   6, 6, ... , 6]]
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
    
    center_x = (tf.sigmoid(boxes[..., 0]) + cell_x) / GRID_SIZE
    center_y = (tf.sigmoid(boxes[..., 1]) + cell_y) / GRID_SIZE
    half_w = anchor_w * tf.exp(boxes[..., 2]) / 2
    half_h = anchor_h * tf.exp(boxes[..., 3]) / 2
    
    corner_x1 = center_x - half_w
    corner_y1 = center_y - half_h
    corner_x2 = center_x + half_w
    corner_y2 = center_y + half_h
    
    return tf.stack([corner_x1, corner_y1, corner_x2, corner_y2], axis=-1)


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
