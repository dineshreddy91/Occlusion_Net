# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a list of path names
"""

from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList


class ListDataset(object):
    def __init__(self, image_lists, transforms=None):
        """
        Initialize the image_lists.

        Args:
            self: (todo): write your description
            image_lists: (todo): write your description
            transforms: (str): write your description
        """
        self.image_lists = image_lists
        self.transforms = transforms

    def __getitem__(self, item):
        """
        Return an image.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        img = Image.open(self.image_lists[item]).convert("RGB")

        # dummy target
        w, h = img.size
        target = BoxList([[0, 0, w, h]], img.size, mode="xyxy")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """
        Returns the length of the image.

        Args:
            self: (todo): write your description
        """
        return len(self.image_lists)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        pass
