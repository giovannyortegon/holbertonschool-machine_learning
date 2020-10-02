#!/usr/bin/env python3
""" Yolo v3 algorithm to perform object detection """
import tensorflow.keras as K
import numpy as np


class Yolo:
    """ Yolo Class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ constructor

        Args:
            model_path:     is the path to where a Darknet Keras model
                            is stored
            classes_path:   is the path to where the list of class names
                            used for the Darknet model, listed in order
                            of index, can be found.
            class_t:        is a float representing the box score threshold
                            for the initial filtering step.
            nms_t:          is a float representing the IOU threshold for
                            non-max suppression.
            anchors:        is a numpy.ndarray of shape (outputs,
                            anchor_boxes, 2) containing all of the
                            anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as classes:
            classes_name = [name.strip() for name in classes]
        self.class_names = classes_name
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors


    def process_outputs(self, outputs, image_size):
        """ process_outputs - 

        Args:
            outputs:    is a list of numpy containing the predictions
                        from the Darknet model for a single image.
            image_size: is a numpy.ndarray containing the image’s
                        original size.

        Returns:
            a tuple of (boxes, box_confidences, box_class_probs)
        """
        box_confidences = list()
        boxes = [output[..., :4] for output in outputs]
        box_class_probs = list()

        image_height, image_width = image_size

        for i in range(len(outputs)):
            grid_height, grid_width, anchor_boxes, _ = outputs[i].shape

        # t_x, t_y, t_w, t_h network outputs
            t_x = outputs[i][..., 0]
            t_y = outputs[i][..., 1]
            t_w = outputs[i][..., 2]
            t_h = outputs[i][..., 3]

        # p_w, p_h anchors dimensions
            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

        # c_x and c_y top-left coordinates of the grid
        # c_y
            c_x = np.tile(np.arange(0, grid_width), grid_height)
            c_x = c_x.reshape(grid_width, grid_width, 1)

        # c_y
            c_y = np.tile(np.arange(0, grid_width), grid_height)
            c_y = c_y.reshape(grid_height, grid_height).T
            c_y = c_y.reshape(grid_height, grid_height, 1)

        # b_x, b_y, b_w, b_h x and y center coordinates, w and h prediction
        b_x = (1 / (1 + np.exp(-t_x))) + c_x
        b_y = (1 / (1 + np.exp(-t_y))) + c_y
        b_w = p_w * np.exp(t_w)
        b_h = p_h * np.exp(t_h)

        # Input size normalize
        b_x /= grid_width
        b_y /= grid_height

        input_width = self.model.input.shape[1].value
        input_height = self.model.input.shape[2].value

        b_w /= input_width
        b_h /= input_height

        # Bonding Box
        x1 = b_x - b_w / 2
        y1 = b_y - b_h / 2
        x2 = b_x + b_w / 2
        y2 = b_y + b_h / 2

        boxes[i][..., 0] = x1 * image_width
        boxes[i][..., 1] = y1 * input_height
        boxes[i][..., 2] = x2 * image_width
        boxes[i][..., 3] = y2 * input_height

        confidences = 1 / (1 + np.exp(-outputs[i][:, :, :, 4]))
        box_confidences.append(confidences.reshape(grid_height,
                                                   grid_width,
                                                   anchor_boxes, 1))

        class_probs = 1 / (1 + np.exp(-outputs[i][:, :, :, 5:]))
        box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
