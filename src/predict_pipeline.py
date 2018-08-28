import numpy as np
import os
import io
import time

from .postprocess_pipeline import PostprocessPipeline
from .io import load_image, save_label, save_score_label, save_class_label
from .constants import MODEL_DIM, YOLO1_CLASS, LABEL_FORMAT
from .preprocess import ResizeTransform

class PredictPipleline():
    def  __init__(self, model, anchors, conf_threshold=0.6, iou_threshold=0.5, dim=MODEL_DIM):
        self.dim = dim
        self.model = model
        self.resize_transform = ResizeTransform(dim, dim)
        self.postpipeline = PostprocessPipeline(anchors, conf_threshold, iou_threshold)

    def predict_one(self, image, normalized=True, show_time=False):
        image_shape = image.shape[:2]
        resized = self.resize_transform.transform(image)
        resized = np.expand_dims(resized, axis=0)

        if normalized:
            resized = resized / 255

        labels = self._predict(resized, show_time)
        labels = np.array(labels[0])
        labels[:, 2:] = self.resize_transform.reverse_transform_bbox(labels[:, 2:], image_shape)
        
        return labels


    def predict_batch(self, image_dir, out_dir, normalized=True, batch_size=32, callback=None, 
        format=LABEL_FORMAT["DEFAULT"], classes=YOLO1_CLASS):
        generator = self._generator(image_dir, batch_size, normalized)

        counter = 1
        while True:
            try:
                images, image_shapes, image_names = next(generator)
            except StopIteration:
                break

            labels = self._predict(images)

            for label, image_shape, image_name in zip(labels, image_shapes, image_names):
                if label:
                    label = np.array(label)
                    label[:, 2:] = self.resize_transform.reverse_transform_bbox(label[:, 2:], image_shape)

                if format == LABEL_FORMAT["DEFAULT"]:
                    save_label(label, os.path.join(out_dir, image_name.replace(".jpg", ".txt")))
                elif format == LABEL_FORMAT["SCORE"]:
                    save_score_label(label, os.path.join(out_dir, image_name.replace(".jpg", ".txt")))
                elif format == LABEL_FORMAT["CLASS"]:
                    save_class_label(label, image_shape, classes, os.path.join(out_dir, image_name.replace(".jpg", ".txt")))

            if callable:
                callback(counter)

            counter += 1


    def _predict(self, X, show_time=False):
        """
        Arguments:
            X: [None, MODEL_DIM, MODEL_DIM, 3]
        Returns:
            labels: [None,  ] - [Prob, class_idx, cx, cy, w, h]
                    cx, cy, w, h is relative to MODEL_DIM
        """
        t0 = time.time()

        Y_predicts = self.model.predict(X)

        t1 = time.time()
        res = []
        for i in range(len(Y_predicts)):
            scores, boxes, classes = self.postpipeline.process(Y_predicts[i])
            labels = []
            for score, box, class_idx in zip(scores, boxes, classes):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                labels.append([score, class_idx, x1 + w/2, y1 + h/2, w, h])

            res.append(labels)

        t2 = time.time()

        if show_time:
            print("Predict: {:.2f}ms, Post: {:.2f}ms".format((t1-t0) * 1000, (t2-t1) * 1000))
        return res


    def _generator(self, image_dir, batch_size=32, normalized=True):
        image_names = [name for name in os.listdir(image_dir) if name.endswith(".jpg")]

        for idx in range(0, len(image_names), batch_size):
            images = []
            image_shapes = []
            for name in image_names[idx:idx+batch_size]:
                image = load_image(os.path.join(image_dir, name))
                image_shape = image.shape[:2]
                resized = self.resize_transform.transform(image)

                if normalized:
                    resized = resized / 255

                images.append(resized)
                image_shapes.append(image_shape)

            yield np.array(images), np.array(image_shapes), np.array(image_names[idx:idx+batch_size])

