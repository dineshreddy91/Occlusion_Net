import torch

from .roi_graph_feature_extractors import *
from maskrcnn_benchmark.modeling.roi_heads.keypoint_head.roi_keypoint_predictors import make_roi_keypoint_predictor
from maskrcnn_benchmark.modeling.roi_heads.keypoint_head.roi_keypoint_feature_extractors import make_roi_keypoint_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.keypoint_head.inference import make_roi_keypoint_post_processor
from maskrcnn_benchmark.modeling.roi_heads.keypoint_head.loss import make_roi_keypoint_loss_evaluator
from .roi_graph_predictors import make_roi_graph_predictor
from .inference import make_roi_graph_post_processor
from .loss import make_roi_graph_loss_evaluator
import numpy as np
import pickle

class ROIGraphHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Initialize the graph

        Args:
            self: (todo): write your description
            cfg: (todo): write your description
            in_channels: (int): write your description
        """
        super(ROIGraphHead, self).__init__()
        self.cfg = cfg.clone()
        self.predictor_heatmap = make_roi_keypoint_predictor(
            cfg, cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS[-1])
        self.feature_extractor = make_roi_graph_feature_extractor(cfg)
        self.feature_extractor_heatmap = make_roi_keypoint_feature_extractor(cfg, in_channels)
        self.post_processor = make_roi_graph_post_processor(cfg)
        self.post_processor_heatmap = make_roi_keypoint_post_processor(cfg)
        self.loss_evaluator_heatmap = make_roi_keypoint_loss_evaluator(cfg)
        self.loss_evaluator = make_roi_graph_loss_evaluator(cfg)
        self.edges = cfg.MODEL.ROI_GRAPH_HEAD.EDGES 
        self.KGNN2D = cfg.MODEL.ROI_GRAPH_HEAD.KGNN2D
        self.KGNN3D = cfg.MODEL.ROI_GRAPH_HEAD.KGNN3D

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator_heatmap.subsample(proposals, targets)
        ## heatmaps computation
        x = self.feature_extractor_heatmap(features,proposals)
        kp_logits = self.predictor_heatmap(x)
 
        if x.shape[0] == 0:
              return torch.zeros((0,x.shape[2],3)),proposals,{}


        ## convert heatmap to graph features
        graph_features = heatmaps_to_graph(kp_logits)
        #x = self.feature_extractor(featu)

        for inc,proposals_per_image in enumerate(proposals):
             proposals_per_image = proposals_per_image.convert("xyxy")
             width = proposals_per_image.bbox[:, 2] - proposals_per_image.bbox[:, 0]
             height = proposals_per_image.bbox[:, 3] - proposals_per_image.bbox[:, 1]
             if inc == 0:
                 ratio = width/height
             else:
                 ratio = torch.cat((ratio,width/height))

        edge_logits,KGNN2D,KGNN3D = self.feature_extractor(graph_features,ratio)

        if not self.training:
            output = graph_features
            if self.edges == True:
                result = self.post_processor(graph_features, edge_logits, proposals)
                output = graph_features
            if self.KGNN2D == True:
                result = self.post_processor(KGNN2D, edge_logits, proposals)
                output = KGNN2D
            if self.KGNN3D ==True:
                result = self.post_processor(KGNN3D, edge_logits, proposals)
                output = KGNN3D
            return output, result, {}
       
        ## process groundtruth proposals
        keypoints_gt, valid_vis_all, valid_invis_all = self.loss_evaluator.process_keypoints(proposals)
        valid_all = valid_vis_all+valid_invis_all 

        ## loss computation
        loss_kp = self.loss_evaluator_heatmap(proposals, kp_logits)
        
        if self.edges == True:
           loss_edges = self.loss_evaluator.loss_edges(valid_vis_all, edge_logits)
           loss_dict_all = dict(loss_edges=loss_edges,loss_kp=loss_kp)
        if self.KGNN2D ==True:
           loss_trifocal = self.loss_evaluator.loss_kgnn2d(keypoints_gt,valid_invis_all, KGNN2D)
           loss_dict_all = dict(loss_edges=loss_edges,loss_kp=loss_kp,loss_trifocal=loss_trifocal)
        if self.KGNN3D ==True:
           valid_all = (valid_vis_all+valid_invis_all)*0 +1 
           valid_all[:,-1] = valid_all[:,-1]*0 # dont compute loss in kgnn3d for center point
           valid_all[:,8] = valid_all[:,8]*0 # dont compute the loss for exhaust        
           loss_kgnn3d = self.loss_evaluator.loss_kgnn3d(KGNN2D, valid_all, KGNN3D)
           loss_dict_all = dict(loss_edges=loss_edges,loss_kp=loss_kp,loss_trifocal=loss_trifocal,loss_kgnn3d=loss_kgnn3d)

        return KGNN2D, proposals, loss_dict_all


def build_roi_graph_head(cfg, in_channels):
    """
    Build a list of the head of the given cfg graph.

    Args:
        cfg: (int): write your description
        in_channels: (int): write your description
    """
    return ROIGraphHead(cfg, in_channels)
