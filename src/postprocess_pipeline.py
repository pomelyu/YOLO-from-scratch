import tensorflow as tf
import keras.backend as K
import numpy as np

from .constants import GRID_SIZE, NUM_BOX, NUM_CLASS
from .postprocess import split_bbox, bbox_cell_to_global, filter_bbox_by_scores, non_maximum_supress

class PostprocessPipeline():
    def __init__(self, conf_threshold=0.6, iou_threshold=0.5):
        """ Construct the post process computation graph
        Arguments:
            conf_threshold: the confidence threshold to filter the candidate bounding box
            iou_threshold: the threshold to evaluated overlapped boxes
        """

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.Y = tf.placeholder("float32", shape=(GRID_SIZE, GRID_SIZE, NUM_BOX, 5 + NUM_CLASS))

        box_confidence, boxes, box_class_probs = split_bbox(self.Y)
        boxes = bbox_cell_to_global(boxes)

        # chekck dimension
        tf.assert_equal(box_confidence.shape.as_list(), [GRID_SIZE, GRID_SIZE, NUM_BOX, 1])
        tf.assert_equal(boxes.shape.as_list(), [GRID_SIZE, GRID_SIZE, NUM_BOX, 4])
        tf.assert_equal(box_class_probs.shape.as_list(), [GRID_SIZE, GRID_SIZE, NUM_BOX, NUM_CLASS])
        
        scores, boxes, classes = filter_bbox_by_scores(
            box_confidence, boxes, box_class_probs, threshold=self.conf_threshold)
        scores, boxes, classes = non_maximum_supress(
            scores, boxes, classes, iou_threshold=self.iou_threshold)
        
        self.scores = scores
        self.boxes = boxes
        self.classes = classes

    def set_conf_threshold(self, conf_threshold=0.6):
        # TODO: 
        # self.conf_threshold = conf_threshold
        pass

    def set_iou_threshold(self, iou_threshold=0.5):
        # TODO:
        # self.iou_threshold = iou_threshold
        pass

    def process(self, Y_pred):
        with tf.Session() as sess:
            feed_dict = { self.Y: Y_pred }
            scores, boxes, classes = sess.run([self.scores, self.boxes, self.classes], feed_dict=feed_dict)
        
        return scores, boxes, classes
