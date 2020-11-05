# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def get_idxs(self, idx):
        """
        Return the indexes of the dataset.

        Args:
            self: (todo): write your description
            idx: (int): write your description
        """
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        """
        Return image info.

        Args:
            self: (todo): write your description
            idx: (str): write your description
        """
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)
