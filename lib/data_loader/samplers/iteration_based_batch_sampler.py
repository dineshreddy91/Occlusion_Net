# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        """
        Initialize the sampler.

        Args:
            self: (todo): write your description
            batch_sampler: (todo): write your description
            num_iterations: (int): write your description
            start_iter: (todo): write your description
        """
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        """
        Iterate over the sampler.

        Args:
            self: (todo): write your description
        """
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        """
        Returns the number of num_iterations.

        Args:
            self: (todo): write your description
        """
        return self.num_iterations
