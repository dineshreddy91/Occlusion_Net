from torch import nn

from maskrcnn_benchmark import layers
from ... import registry


@registry.ROI_GRAPH_PREDICTOR.register("graphRCNNPredictor")
class GraphRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            cfg: (todo): write your description
            in_channels: (int): write your description
        """
        super(GraphRCNNPredictor, self).__init__()
    def forward(self, x):
        """
        Forward function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return x


def make_roi_graph_predictor(cfg, in_channels):
    """
    Make a graph of - based on the graph.

    Args:
        cfg: (todo): write your description
        in_channels: (int): write your description
    """
    func = registry.ROI_GRAPH_PREDICTOR[cfg.MODEL.ROI_GRAPH_HEAD.PREDICTOR]
    return func(cfg, in_channels)
