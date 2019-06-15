import torch
from torch.nn import functional as F
import numpy as np
from maskrcnn_benchmark.modeling.matcher import Matcher

from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler,
)
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from maskrcnn_benchmark.structures.keypoint import keypoints_to_heat_map
from sklearn.decomposition import PCA
import sys


def keypoints_scaled(keypoints, rois, heatmap_size , bb_pad):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long(), rois.new().long(), rois.new().long()

    width = rois[:, 2] - rois[:, 0]
    height = rois[:, 3] - rois[:, 1]
    offset_x = rois[:, 0]-width*bb_pad
    offset_y = rois[:, 1]-height*bb_pad
    scale_x = heatmap_size / (width*(1+bb_pad))
    scale_y = heatmap_size / (height*(1+bb_pad))

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]
    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()
    
    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1
    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    #print(keypoints)
    vis = (keypoints[..., 2] > 1)
    invis = keypoints[..., 2] == 1
    valid_vis = (valid_loc & vis).long()
    valid_invis = (valid_loc & invis).long()
    squashed = torch.cat([x.view(-1,x.shape[0],x.shape[1]),y.view(-1,y.shape[0],y.shape[1])])
    return squashed, valid_vis, valid_invis

def keypoints_to_squash(keypoints, proposals, discretization_size, bb_pad):
    proposals = proposals.convert("xyxy")
    return keypoints_scaled(keypoints.keypoints, proposals.bbox, discretization_size, bb_pad)


def cat_boxlist_with_keypoints(boxlists):
    assert all(boxlist.has_field("keypoints") for boxlist in boxlists)

    kp = [boxlist.get_field("keypoints").keypoints for boxlist in boxlists]
    kp = cat(kp, 0)

    fields = boxlists[0].get_fields()
    fields = [field for field in fields if field != "keypoints"]

    boxlists = [boxlist.copy_with_fields(fields) for boxlist in boxlists]
    boxlists = cat_boxlist(boxlists)
    boxlists.add_field("keypoints", kp)
    return boxlists

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def _within_box(points, boxes):
    """Validate which keypoints are contained inside a given box.
    points: NxKx2
    boxes: Nx4
    output: NxK
    """
    x_within = (points[..., 0] >= boxes[:, 0, None]) & (
        points[..., 0] <= boxes[:, 2, None]
    )
    y_within = (points[..., 1] >= boxes[:, 1, None]) & (
        points[..., 1] <= boxes[:, 3, None]
    )
    return x_within & y_within

class KeypointRCNNLossComputation(object):
    def __init__(self, proposal_matcher, fg_bg_sampler, discretization_size, cfg):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.discretization_size = discretization_size
        self.bb_pad = cfg.MODEL.ROI_GRAPH_HEAD.BB_PAD
        off_diag = np.ones([cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES, cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES]) - np.eye(cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES)
        self.idx =  torch.LongTensor(np.where(off_diag)[1].reshape(cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES,cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES-1)).cuda()

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Keypoint RCNN needs "labels" and "keypoints "fields for creating the targets
        target = target.copy_with_fields(["labels", "keypoints"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        keypoints = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            # TODO check if this is the right one, as BELOW_THRESHOLD
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            keypoints_per_image = matched_targets.get_field("keypoints")
            within_box = _within_box(
                keypoints_per_image.keypoints, matched_targets.bbox
            )
            vis_kp = keypoints_per_image.keypoints[..., 2] > 0
            is_visible = (within_box & vis_kp).sum(1) > 0

            labels_per_image[~is_visible] = -1

            labels.append(labels_per_image)
            keypoints.append(keypoints_per_image)

        return labels, keypoints

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, keypoints = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, keypoints_per_image, proposals_per_image in zip(
            labels, keypoints, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("keypoints", keypoints_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    #def __call__(self, proposals, keypoint_logits):
    def process_keypoints(self, proposals):
        for inc, proposals_per_image in enumerate(proposals):
            kp = proposals_per_image.get_field("keypoints")
            keypoint_squashed = kp.keypoints
            keypoints_per_image, valid_vis, valid_invis = keypoints_to_squash(kp, proposals_per_image, self.discretization_size, self.bb_pad)
            if keypoints_per_image.shape[0] != 0:
                keypoints_per_image = keypoints_per_image.permute(1, 2, 0)

                if inc == 0:
                   keypoints_gt = keypoints_per_image
                   vis_all = valid_vis
                   invis_all = valid_invis
                else:
                   keypoints_gt = torch.cat((keypoints_gt,keypoints_per_image))
                   vis_all = torch.cat((vis_all,valid_vis))
                   invis_all = torch.cat((invis_all,valid_invis))
        return keypoints_gt,vis_all,invis_all

    def loss_kgnn2d(self, keypoints_gt, valid_points, keypoints_logits):
        keypoints_gt = keypoints_gt.type(torch.FloatTensor)*valid_points.unsqueeze(2).type(torch.FloatTensor)
        keypoints_logits = keypoints_logits.type(torch.FloatTensor)*valid_points.unsqueeze(2).type(torch.FloatTensor)
        keypoints_gt = keypoints_gt.cuda()
        keypoints_logits = keypoints_logits.cuda()
        loss_occ = nll_gaussian(keypoints_gt[:,:,0:2], keypoints_logits[:,:,0:2] , 3)
        return loss_occ


    def loss_edges(self, valid_points, edges):
        relations = torch.zeros(valid_points.shape[0],valid_points.shape[1]*(valid_points.shape[1]-1)).cuda()
        for count,vis in enumerate(valid_points):
            vis = vis.view(-1,1)
            vis = vis*vis.t()
            vis = torch.gather(vis,1,self.idx)
            relations[count] = vis.view(-1)
        relations = relations.type(torch.LongTensor).cuda() 
        loss_edges = F.cross_entropy(edges.view(-1, 2), relations.view(-1))
        return loss_edges



    def loss_edges_old(self, valid_points, edges):
        relations = torch.zeros(valid_points.shape[0],valid_points.shape[1]*(valid_points.shape[1]-1))
        count = 0
        for vis in valid_points:
            adj_mat = torch.zeros(relations.shape[1]).cuda()
            
            loop=0
            for i in range(vis.shape[0]):
                for j in range(vis.shape[0]):
                    if i == j:
                       continue
                    if vis[i] == 1 and vis[j] == 1:
                       adj_mat[loop] = 0
                    else:
                       adj_mat[loop] = 1
                    loop = loop+1
            
            print(adj_mat)
#            print(relations[count].shape,adj_mat.shape)
            relations[count] = adj_mat  
        relations = relations.type(torch.LongTensor).cuda() 
        loss_edges = F.cross_entropy(edges.view(-1, 2), relations.view(-1))
        return loss_edges


    def loss_kgnn3d(self, keypoint_kgnn2d, valid, projected_points):
        keypoint_kgnn2d = keypoint_kgnn2d.type(torch.FloatTensor).cuda()
        projected_points = projected_points.type(torch.FloatTensor).cuda()
        keypoint_kgnn2d = keypoint_kgnn2d*valid.unsqueeze(2).type(torch.FloatTensor).cuda()
        projected_points = projected_points*valid.unsqueeze(2).type(torch.FloatTensor).cuda()
        #print(projected_points[0,:,0:2])
        #print(keypoint_kgnn2d[0,:,0:2])
        loss_kgnn3d = nll_gaussian(keypoint_kgnn2d[:,:,0:2], projected_points[:,:,0:2] , 100)
        if torch.isnan(loss_kgnn3d):
              sys.exit("kgnn3d error exploded")
        #print(loss_kgnn3d)
        return loss_kgnn3d

 
def make_roi_graph_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )
    resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION
    loss_evaluator = KeypointRCNNLossComputation(matcher, fg_bg_sampler, resolution, cfg)
    return loss_evaluator
