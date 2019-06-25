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
            print(graph_features[0])
            print(KGNN2D[0])
            #print(edge_logits[0])
            #asas
            result = self.post_processor(KGNN3D, edge_logits, proposals)
            print(result[0])
            result = self.post_processor(KGNN2D, edge_logits, proposals)
            result = self.post_processor(graph_features, edge_logits, proposals)
            return KGNN2D, result, {}


       
        ## process groundtruth proposals
        keypoints_gt, valid_vis_all, valid_invis_all = self.loss_evaluator.process_keypoints(proposals)
        #print(valid_vis_all)
        #print(valid_invis_all)
        valid_all = valid_vis_all+valid_invis_all 
        #print(valid_all)

        ## loss computation
        loss_kp = self.loss_evaluator_heatmap(proposals, kp_logits)
        #print(valid_all)
        loss_edges = self.loss_evaluator.loss_edges(valid_vis_all, edge_logits)
        loss_trifocal = self.loss_evaluator.loss_kgnn2d(keypoints_gt,valid_all, KGNN2D)
        #print(rtb_KGNN3D)
        valid_all = (valid_vis_all+valid_invis_all)*0 +1 
        valid_all[:,-1] = valid_all[:,-1]*0 # dont compute loss in kgnn3d for center point
        valid_all[:,8] = valid_all[:,8]*0 # dont compute the loss for exhaust
        loss_kgnn3d = self.loss_evaluator.loss_kgnn3d(KGNN2D, valid_all, KGNN3D)
        #loss_kgnn3d = self.loss_evaluator.loss_kgnn3d(KGNN3D, valid_all, KGNN2D)
        #return KGNN2D, proposals, dict(loss_edges=loss_edges,loss_kp=loss_kp,loss_kgnn2d=loss_kgnn2d)
        return KGNN2D, proposals, dict(loss_edges=loss_edges,loss_kp=loss_kp,loss_trifocal=loss_trifocal,loss_kgnn3d=loss_kgnn3d)
        #return KGNN2D, proposals, dict(loss_kgnn2d=loss_kgnn2d, loss_edges=loss_edges)


def build_roi_graph_head(cfg, in_channels):
    return ROIGraphHead(cfg, in_channels)
