#!/usr/bin/env python3
""" Yolo v3 algorithm to perform object detection """
import tensorflow.keras as K


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
