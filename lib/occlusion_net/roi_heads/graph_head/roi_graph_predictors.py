from torch import nn

from maskrcnn_benchmark import layers
from ... import registry


@registry.ROI_GRAPH_PREDICTOR.register("graphRCNNPredictor")
class GraphRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(GraphRCNNPredictor, self).__init__()
    def forward(self, x):
        return x


def make_roi_graph_predictor(cfg, in_channels):
    func = registry.ROI_GRAPH_PREDICTOR[cfg.MODEL.ROI_GRAPH_HEAD.PREDICTOR]
    return func(cfg, in_channels)
