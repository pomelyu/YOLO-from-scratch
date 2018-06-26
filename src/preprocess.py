import numpy as np
from PIL import Image

def preprocess_image(image, model_dim):
    """ Resize image to required dimension and add necessary padding

    Arguments:
        image: PIL image object
        model_dim: the required dimension
    """
    img_size = image.size
    ratio = min(model_dim / img_size[0], model_dim / img_size[1])
    new_w = int(img_size[0] * ratio)
    new_h = int(img_size[1] * ratio)
    resized = image.resize((new_w, new_h), Image.ANTIALIAS)

    padded = Image.new("RGB", (model_dim, model_dim))
    padded.paste(resized, box=((model_dim-new_w) // 2, (model_dim-new_h) // 2))
    
    return padded

def generate_bboxs(labels, image, model_dim, grid_size, num_box, num_class):
    """ Divide image into grids and generate target value for YOLO training

    Arguments:
        labels: labels from text file
        image: the original image
        model_dim: the required dimension
        grid_size: the size of grids
        num_box: maximum bounding box(objects) could be described by a cell
        num_class: number of object class

    Returns:
        bboxs: the numpy array with (grid_size, grid_size, num_box*(5+num_class))
    """
    img_size = image.size
    ratio = min(model_dim / img_size[0], model_dim / img_size[1])
    new_w = int(img_size[0] * ratio)
    new_h = int(img_size[1] * ratio)

    obj_in_cells = {}
    for label in labels:
        # align the dimension
        bx = (label[1] * new_w + (model_dim - new_w) // 2) / model_dim
        by = (label[2] * new_h + (model_dim - new_h) // 2) / model_dim
        bw = (label[3] * new_w) / model_dim
        bh = (label[4] * new_h) / model_dim

        # calculate the cell which this bbox belong to
        grid_x = bx * grid_size
        grid_y = by * grid_size

        cell_idx = (int(grid_x), int(grid_y))

        obj = (label[0], grid_x - cell_idx[0], grid_y - cell_idx[1], bw, bh)
        if cell_idx not in obj_in_cells:
            obj_in_cells[cell_idx] = [obj]
        else:
            obj_in_cells[cell_idx].append(obj)

    box_size = 5 + num_class
    bboxs = np.zeros((grid_size, grid_size, num_box * box_size))
    for (i, j), objs in obj_in_cells.items():
        for idx, bbox in enumerate(objs):
            if idx > 1: break

            bboxs[i][j][idx * box_size] = 1
            bboxs[i][j][idx * box_size + 1] = bbox[1]
            bboxs[i][j][idx * box_size + 2] = bbox[2]
            bboxs[i][j][idx * box_size + 3] = bbox[3]
            bboxs[i][j][idx * box_size + 4] = bbox[4]
            bboxs[i][j][idx * box_size + 5 + bbox[0]] = 1

    return bboxs
