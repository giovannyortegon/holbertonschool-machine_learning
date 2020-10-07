#!/usr/bin/env python3
""" Yolo v3 algorithm to perform object detection """
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import cv2
import glob


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
        """ process_outputs

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
            b_x = (1. / (1. + np.exp(-t_x))) + c_x
            b_y = (1. / (1. + np.exp(-t_y))) + c_y
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
            x1 = (b_x - b_w / 2)
            y1 = (b_y - b_h / 2)
            x2 = (b_x + b_w / 2)
            y2 = (b_y + b_h / 2)

            boxes[i][..., 0] = x1 * image_width
            boxes[i][..., 1] = y1 * image_height
            boxes[i][..., 2] = x2 * image_width
            boxes[i][..., 3] = y2 * image_height

            confidences = 1 / (1 + np.exp(-outputs[i][:, :, :, 4]))
            box_confidences.append(confidences.reshape(grid_height,
                                                       grid_width,
                                                       anchor_boxes, 1))

            class_probs = 1 / (1 + np.exp(-outputs[i][:, :, :, 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ filter_boxes

        Args:
            boxes:  a list of numpy.ndarrays containing the
                    processed boundary boxes for each output,
                    respectively.
            box_confidences:    a list of numpy.ndarrays containing
                                the processed box confidences for
                                each output, respectively.
            box_class_probs:    a list of numpy.ndarrays containing
                                the processed box class probabilities
                                for each output, respectively.

        Returns:
            a tuple of (filtered_boxes, box_classes, box_scores)
        """
        boxes_score = [i * j for i, j in zip(box_confidences, box_class_probs)]

        classes = [score.argmax(axis=-1) for score in boxes_score]
        class_reshape = [cls.reshape(-1) for cls in classes]
        box_class = np.concatenate(class_reshape, axis=-1)

        scores = [score.max(axis=-1) for score in boxes_score]
        score_reshape = [score.reshape(-1) for score in scores]
        box_class_score = np.concatenate(score_reshape, axis=-1)

        boxes_reshape = [box.reshape(-1, 4) for box in boxes]
        boxes_processed = np.concatenate(boxes_reshape, axis=0)

        filter_mask = np.where(box_class_score >= self.class_t)

        filtered_boxes = boxes_processed[filter_mask]
        box_classes = box_class[filter_mask]
        box_scores = box_class_score[filter_mask]

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ non_max_suppression

        Args:
            filtered_boxes: a numpy array containing all of the
                            filtered bounding boxes.
            box_classes:    a numpy array containing the class number for
                            the class that filtered_boxes predicts,
                            respectively.
            box_scores:     a numpy array containing the box scores for
                            each box in filtered_boxes, respectively.

        Returns:
            a tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores).
        """
        box = list()
        box_class = list()
        box_score = list()

        for bx in np.unique(box_classes):
            indices = np.where(box_classes == bx)
            boxes = filtered_boxes[indices]
            boxes_class = box_classes[indices]
            boxes_score = box_scores[indices]

            box_x1 = boxes[:, 0]
            box_y1 = boxes[:, 1]
            box_x2 = boxes[:, 2]
            box_y2 = boxes[:, 3]

            box_area = (box_x2 - box_x1 + 1) * (box_y2 - box_y1 + 1)
            order = boxes_score.argsort()[::-1]

            inds = list()

            while order.size > 0:
                i = order[0]
                inds.append(i)
                xi1 = np.maximum(box_x1[i], box_x1[order[1:]])
                yi1 = np.maximum(box_y1[i], box_y1[order[1:]])
                xi2 = np.minimum(box_x2[i], box_x2[order[1:]])
                yi2 = np.minimum(box_y2[i], box_y2[order[1:]])
                inter_width = np.maximum(0.0, xi2 - xi1 + 1)
                inter_height = np.maximum(0.0, yi2 - yi1 + 1)
                inter_area = inter_width * inter_height
                union_area = (box_area[i] + box_area[order[1:]]) - inter_area
                iou = inter_area / union_area

                nms_indices = np.where(iou <= self.nms_t)[0]
                order = order[nms_indices + 1]

            box.append(boxes[inds])
            box_class.append(boxes_class[inds])
            box_score.append(boxes_score[inds])

        box_predictions = np.concatenate(box, axis=0)
        predicted_box_classes = np.concatenate(box_class, axis=0)
        predicted_box_scores = np.concatenate(box_score, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """ load_images

        Args:
            folder_path:    a string representing the path to the folder
                            holding all the images to load.

        Returns:
            a tuple of (images, image_paths).
        """
        images = list()
        image_paths = list()

        for image in glob.glob(folder_path + "/*.*"):
            image_paths.append(image)
            images.append(cv2.imread(image))

        return images, image_paths

    def preprocess_images(self, images):
        """ preprocess_images

        Args:
            images: a list of images as numpy array

        Return:
            a tuple of (pimages, image_shapes)
        """
        input_width = self.model.input.shape[1].value
        input_height = self.model.input.shape[2].value
        dim = (input_width, input_height)

        pimages = list()
        image_shapes = list()

        for img in images:
            process = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
            pimages.append(process / 255)
            image_shapes.append(img.shape[:2])

        return np.array(pimages), np.array(image_shapes)
