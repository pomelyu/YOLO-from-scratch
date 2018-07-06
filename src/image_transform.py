import numpy as np

def flip_image_horizontal(image, bbox):
    image = np.flip(image, axis=1)
    bbox = np.flip(bbox, axis=0)
    bbox[:, :, :, 1] = 1 - bbox[:, :, :, 1]
    
    return image, bbox


def covert_to_VGG_input(image):
    VGG_MEAN = [103.939, 116.779, 123.68]
    
    image = image[:, :, ::-1]
    image[:, :, 0] = image[:, :, 0] - VGG_MEAN[0]
    image[:, :, 1] = image[:, :, 1] - VGG_MEAN[1]
    image[:, :, 2] = image[:, :, 2] - VGG_MEAN[2]
    
    return image
