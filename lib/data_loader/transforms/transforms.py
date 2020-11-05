# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        """
        Initialize the transport.

        Args:
            self: (todo): write your description
            transforms: (str): write your description
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """
        Returns a tuple of an image.

        Args:
            self: (todo): write your description
            image: (array): write your description
            target: (int): write your description
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        """
        Return a human - readable representation of this object.

        Args:
            self: (todo): write your description
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        """
        Initialize the min_size.

        Args:
            self: (todo): write your description
            min_size: (int): write your description
            max_size: (int): write your description
        """
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        """
        Get the size of a image.

        Args:
            self: (todo): write your description
            image_size: (int): write your description
        """
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        """
        Resizes the image.

        Args:
            self: (todo): write your description
            image: (array): write your description
            target: (todo): write your description
        """
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        """
        Initialize the properties.

        Args:
            self: (todo): write your description
            prob: (todo): write your description
        """
        self.prob = prob

    def __call__(self, image, target):
        """
        Transpose the image.

        Args:
            self: (todo): write your description
            image: (array): write your description
            target: (todo): write your description
        """
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        """
        Call an image with the given image.

        Args:
            self: (todo): write your description
            image: (array): write your description
            target: (int): write your description
        """
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        """
        Initialize the bgr.

        Args:
            self: (todo): write your description
            mean: (float): write your description
            std: (array): write your description
            to_bgr255: (todo): write your description
        """
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        """
        Wrapper for the image

        Args:
            self: (todo): write your description
            image: (array): write your description
            target: (int): write your description
        """
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
