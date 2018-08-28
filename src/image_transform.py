import numpy as np
from skimage import exposure
from skimage import transform

def histogram_equalization(image):
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    return image

def flip_image_horizontal(image, label):
    image = image[:, ::-1, :]
    label[:, 1] = 1 - label[:, 1] # cx
    return image, label


def adjust_gamma(image, gamma):
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    image = exposure.adjust_gamma(image, gamma)
    
    return image

def translation_and_scale_image(image, label, s, dx, dy):
    h, w , _= image.shape

    # translate to left-top (inverse)
    mat_trans_ori = transform.AffineTransform(translation=(w/2, h/2))
    # perform scale s (inverse)
    mat_scale = transform.AffineTransform(scale=(1/s, 1/s))
    # translate back to center (inverse)
    mat_trans_cent = transform.AffineTransform(translation=(-w/2, -h/2))
    # perfom translate dx, dy (inverse)
    mat_trans = transform.AffineTransform(translation=(-dx, -dy))

    mat = np.dot(np.dot(np.dot(mat_trans_ori.params, mat_scale.params), mat_trans_cent.params), mat_trans.params)

    image = transform.warp(image, inverse_map=mat)
    label[:, 1] = (label[:, 1] - 0.5) * s + 0.5 + dx / w
    label[:, 2] = (label[:, 2] - 0.5) * s + 0.5 + dy / h
    label[:, 3] = label[:, 3] * s
    label[:, 4] = label[:, 4] * s

    res_label = []
    for lb in label:
        if lb[1] < 0 or lb[1] > 1 or lb[2] < 0 or lb[2] > 1:
            continue

        half_w = lb[3] / 2
        half_h = lb[4] / 2
        if lb[1] - half_w < 0:
            lb[3] = lb[1] + half_w
            lb[1] = lb[3] / 2
        elif lb[1] + half_w > 1:
            lb[3] = 1 - (lb[1] - half_w)
            lb[1] = 1 - lb[3] / 2

        if lb[2] - half_h < 0:
            lb[4] = lb[2] + half_h
            lb[2] = lb[4] / 2
        elif lb[2] + half_h > 1:
            lb[4] = 1 - (lb[2] - half_h)
            lb[2] = 1 - lb[4] / 2

        res_label.append(lb)

    return image, np.array(res_label)

def covert_to_VGG_input(image):
    VGG_MEAN = [103.939, 116.779, 123.68]
    
    image = image[:, :, ::-1]
    image[:, :, 0] = image[:, :, 0] - VGG_MEAN[0]
    image[:, :, 1] = image[:, :, 1] - VGG_MEAN[1]
    image[:, :, 2] = image[:, :, 2] - VGG_MEAN[2]
    
    return image
