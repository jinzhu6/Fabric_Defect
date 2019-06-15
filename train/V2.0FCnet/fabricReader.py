"""Reader for Cityscape dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import paddle.dataset as dataset

DATA_PATH = "./trainData/20181024/out"
TRAIN_LIST = DATA_PATH + "/train.list"
TEST_LIST = DATA_PATH + "/val.list"
IGNORE_LABEL = 255
NUM_CLASSES = 10
TRAIN_DATA_SHAPE = (3, 1200, 2448)
TEST_DATA_SHAPE = (3, 1200, 2448)
IMG_MEAN = np.array((0, 0, 0), dtype=np.float32)






def train_data_shape():
    return TRAIN_DATA_SHAPE


def num_classes():
    return NUM_CLASSES


class DataGenerater:
    def __init__(self, data_list, flip=False, scaling=False):
        self.flip = flip
        self.scaling = scaling
        self.image_label = []
        with open(data_list, 'r') as f:
            for line in f:
                image_file, label_file = line.strip().split(' ')
                self.image_label.append((image_file, label_file))

    def create_train_reader(self):
        """
        Create a reader for train dataset.
        """

        def reader():
            for image, label in self.image_label:
                image, label_sub1 = self.process_train_data(
                    image, label)
                yield self.mask(
                    np.array(image),
                    np.array(label_sub1))

        return reader

    def process_train_data(self, image, label):
        """
        Process training data.
        """
        image, label = self.load(image, label)
        if self.flip:
            image, label = self.random_flip(image, label)
        if self.scaling:
            image, label = self.random_scaling(image, label)
        image, label = self.resize(image, label, out_size=TRAIN_DATA_SHAPE[1:])
        label = label.astype("float32")
        label_sub = self.scale_label(label, factor=4)
        image = dataset.image.to_chw(image)
        return image, label_sub

    def load(self, image, label):
        """
        Load image from file.
        """
        image = dataset.image.load_image(
            DATA_PATH + "/" + image, is_color=True).astype("float32")
        image -= IMG_MEAN
        label = dataset.image.load_image(
            DATA_PATH + "/" + label, is_color=False).astype("float32")
        return image, label

    def random_flip(self, image, label):
        """
        Flip image and label randomly.
        """
        r = np.random.rand(1)
        if r > 0.5:
            image = dataset.image.left_right_flip(image, is_color=True)
            label = dataset.image.left_right_flip(label, is_color=False)
        return image, label

    def random_scaling(self, image, label):
        """
        Scale image and label randomly.
        """
        scale = np.random.uniform(0.5, 2.0, 1)[0]
        h_new = int(image.shape[0] * scale)
        w_new = int(image.shape[1] * scale)
        image = cv2.resize(image, (w_new, h_new))
        label = cv2.resize(
            label, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        return image, label

    def padding_as(self, image, h, w, is_color):
        """
        Padding image.
        """
        pad_h = max(image.shape[0], h) - image.shape[0]
        pad_w = max(image.shape[1], w) - image.shape[1]
        if is_color:
            return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        else:
            return np.pad(image, ((0, pad_h), (0, pad_w)), 'constant')

    def random_crop(self, im, out_shape, is_color=True):
        h, w = im.shape[:2]
        h_start = np.random.randint(0, h - out_shape[0] + 1)
        w_start = np.random.randint(0, w - out_shape[1] + 1)
        h_end, w_end = h_start + out_shape[0], w_start + out_shape[1]
        if is_color:
            im = im[h_start:h_end, w_start:w_end, :]
        else:
            im = im[h_start:h_end, w_start:w_end]
        return im

    def resize(self, image, label, out_size):
        """
        Resize image and label by padding or cropping.
        """
        ignore_label = IGNORE_LABEL
        label = label - ignore_label
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        combined = np.concatenate((image, label), axis=2)
        combined = self.padding_as(
            combined, out_size[0], out_size[1], is_color=True)
        combined = self.random_crop(combined, out_size, is_color=True)
        image = combined[:, :, 0:3]
        label = combined[:, :, 3:4] + ignore_label
        return image, label

    def scale_label(self, label, factor):
        """
        Scale label according to factor.
        """
        h = label.shape[0] // factor
        w = label.shape[1] // factor
        return cv2.resize(
            label, (h, w), interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]

    def mask(self, image, label0):
        """
        Get mask for valid pixels.
        """
        mask_sub1 = np.where(((label0 < (NUM_CLASSES + 1)) & (
                label0 != IGNORE_LABEL)).flatten())[0].astype("int32")
        shape1 = [1]
        for i in image.shape:
            shape1.append(i)
        shape3 = [1]
        for i in label0.shape:
            shape3.append(i)
        image = image.astype("float32").reshape(shape1)
        label0 = label0.reshape(shape3)
        mask_sub1=mask_sub1.reshape(1,-1)
        image = image / 255.0 * 2.0 - 1.0
        label0 = label0 / 255.0 * 2.0 - 1.0
        return image, label0


def train(flip=True, scaling=True):
    """
    Cityscape training set reader.
    It returns a reader, in which each result is a batch with batch_size samples.

    :param batch_size: The batch size of each result return by the reader.
    :type batch_size: int
    :param flip: Whether flip images randomly.
    :type batch_size: bool
    :param scaling: Whether scale images randomly.
    :type batch_size: bool
    :return: Training reader.
    :rtype: callable
    """
    reader = DataGenerater(
        TRAIN_LIST, flip=flip, scaling=scaling).create_train_reader()
    return reader
