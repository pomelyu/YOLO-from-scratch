import numpy as np
import os
import io

from .postprocess_pipeline import PostprocessPipeline
from .io import load_image, save_label, save_score_label
from .constants import MODEL_DIM
from .preprocess import preprocess_image, restore_label

class PredictPipleline():
    def  __init__(self, model, anchors, conf_threshold=0.6, iou_threshold=0.5, dim=MODEL_DIM):
        self.dim = dim
        self.model = model
        self.postpipeline = PostprocessPipeline(anchors, conf_threshold, iou_threshold)

    def predict_one(self, image, normalized=True):
        image_shape = image.shape[:2]
        resized = preprocess_image(image, self.dim)
        resized = np.expand_dims(resized, axis=0)

        if normalized:
            resized = resized / 255

        labels = self._predict(resized)
        label = restore_label(labels[0], image_shape, self.dim)
        return label


    def predict_batch(self, image_dir, out_dir, normalized=True, batch_size=32, callback=None, with_score=False):
        generator = self._generator(image_dir, batch_size, normalized)

        while True:
            try:
                images, image_shapes, image_names = next(generator)
            except StopIteration:
                break

            labels = self._predict(images)

            for label, image_shape, image_name in zip(labels, image_shapes, image_names):
                label = restore_label(label, image_shape, self.dim)
                if with_score:
                    save_score_label(label, os.path.join(out_dir, image_name.replace(".jpg", ".txt")))
                else:
                    save_label(label, os.path.join(out_dir, image_name.replace(".jpg", ".txt")))

            if callable:
                callback()


    def _predict(self, X):
        """
        Arguments:
            X: [None, MODEL_DIM, MODEL_DIM, 3]
        Returns:
            labels: [None,  ] - [Prob, class_idx, cx, cy, w, h]
                    cx, cy, w, h is relative to MODEL_DIM
        """
        Y_predicts = self.model.predict(X)
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

        return res


    def _generator(self, image_dir, batch_size=32, normalized=True):
        image_names = [name for name in os.listdir(image_dir) if name.endswith(".jpg")]
        
        for idx in range(0, len(image_names), batch_size):
            images = []
            image_shapes = []
            for name in image_names[idx:idx+batch_size]:
                image = load_image(os.path.join(image_dir, name))
                image_shape = image.shape[:2]
                resized = preprocess_image(image, self.dim)

                if normalized:
                    resized = resized / 255

                images.append(resized)
                image_shapes.append(image_shape)

            yield np.array(images), np.array(image_shapes), np.array(image_names[idx:idx+batch_size])

