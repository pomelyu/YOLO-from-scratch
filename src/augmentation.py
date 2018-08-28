import numpy as np

from .constants import GAMMA_MIN, GAMMA_MAX, TRANS_MIN, TRANS_MAX, SCALE_MIN, SCALE_MAX
from .image_transform import flip_image_horizontal, adjust_gamma, translation_and_scale_image

class Augmentation():
    def __init__(self, flip=True, gamma=True, translate_and_scale=True):
        self.flip = flip
        self.gamma = gamma
        self.translate_and_scale = translate_and_scale

    def transform(self, image, label):
        if self.flip and np.random.uniform(0, 1) > 0.5:
            image, label = flip_image_horizontal(image, label)

        if self.gamma:
            gamma = np.random.uniform(GAMMA_MIN, GAMMA_MAX)
            image = adjust_gamma(image, gamma)

        if self.translate_and_scale:
            s = np.random.uniform(SCALE_MIN, SCALE_MAX)
            dx = np.random.uniform(TRANS_MIN, TRANS_MAX)
            dy = np.random.uniform(TRANS_MIN, TRANS_MAX)

            image, label = translation_and_scale_image(image, label, s, dx, dy)

        return image, label
