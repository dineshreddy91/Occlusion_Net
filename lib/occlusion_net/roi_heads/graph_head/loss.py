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


def hamilton_product(qa, qb):
    """Multiply qa by qb.
    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]
    
    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]
    
    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0
    
    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)

def quat_rotate(X, q):
    """Rotate points by quaternions.
    Args:
        X: B X N X 3 points
        q: B X 4 quaternions
    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    ones_x = X[[0], :, :][:, :, [0]]*0 + 1
    q = torch.unsqueeze(q, 1)*ones_x

    q_conj = torch.cat([ q[:, :, [0]] , -1*q[:, :, 1:4] ], dim=-1)
    X = torch.cat([ X[:, :, [0]]*0, X ], dim=-1)
    
    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]


def orthographic_proj_withz(X, cam, ratio, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    
    X_rot = quat_rotate(X, quat)

    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)

    proj = scale * X_rot

    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z
    
    ratio = ratio.repeat(14,1).permute(1, 0).contiguous().view(-1,14,1)
    proj_xy = proj_xy*torch.cat((ratio*0+1, ratio), 2)#.view(-1,12)
    return torch.cat((proj_xy, proj_z), 2)


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
    vis = keypoints[..., 2] == 1
    invis = keypoints[..., 2] == 0
    valid_vis = (valid_loc & vis).long()
    valid_invis = (valid_loc & invis).long()
    squashed = torch.cat([x.view(-1,x.shape[0],x.shape[1]),y.view(-1,y.shape[0],y.shape[1])])
    ratio = width/height
    return squashed, valid_vis, valid_invis, ratio

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

def pca_computation(path): 
        pca_3d = np.load(path)
        first_half = pca_3d[:,0:8,:]
        second_half = pca_3d[:,8:,:]
        sample_rows = np.zeros((pca_3d.shape[0],1,pca_3d.shape[2]))
        pca_3d = np.concatenate((first_half,sample_rows), axis=1)
        pca_3d = np.concatenate((pca_3d,second_half), axis=1)
        pca_3d = np.concatenate((pca_3d,sample_rows), axis=1)
        Xtrain = np.zeros((pca_3d.shape[0],21))
        for inc, pca_inc in enumerate(pca_3d):
            imp_points = pca_inc[::2,3:6]
        
            imp_points = imp_points.flatten()
            Xtrain[inc,:] = imp_points
    
        pca = PCA(n_components=5)
        pca.fit(Xtrain)

        U, S, VT = np.linalg.svd(Xtrain - Xtrain.mean(0))
        X_train_pca = pca.transform(Xtrain)
    
        mean_shape=torch.FloatTensor(pca.mean_)
        pca_component = torch.FloatTensor(pca.components_)
        mean_shape, pca_component = mean_shape.cuda(), pca_component.cuda()
        return mean_shape, pca_component


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
        self.mean_shape, self.pca_component = pca_computation('data/pca_3d_cad.npy')
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
        loop = 0
        for proposals_per_image in proposals:
            kp = proposals_per_image.get_field("keypoints")
            keypoint_squashed = kp.keypoints
            keypoints_per_image, valid_vis, valid_invis,ratio = keypoints_to_squash(kp, proposals_per_image, self.discretization_size, self.bb_pad)
            if keypoints_per_image.shape[0] == 0:
                 continue
            keypoints_per_image = keypoints_per_image.permute(1, 2, 0)

            if loop == 0:
                   keypoints_gt = keypoints_per_image
                   valid_vis_all = valid_vis
                   valid_invis_all = valid_invis
                   ratio_all = ratio
            else:
                   keypoints_gt = torch.cat((keypoints_gt,keypoints_per_image))
                   valid_vis_all = torch.cat((valid_vis_all,valid_vis))
                   valid_invis_all = torch.cat((valid_invis_all,valid_invis))
                   ratio_all = torch.cat((ratio_all,ratio))
            loop = loop+1
        return keypoints_gt,valid_vis_all,valid_invis_all, ratio_all

    def loss_kgnn2d(self, keypoints_gt, valid_points, keypoints_logits):
        keypoints_gt = keypoints_gt.type(torch.FloatTensor)*valid_points.unsqueeze(2).type(torch.FloatTensor)
        keypoints_logits = keypoints_logits.type(torch.FloatTensor)*valid_points.unsqueeze(2).type(torch.FloatTensor)
        keypoints_gt = keypoints_gt.cuda()
        keypoints_logits = keypoints_logits.cuda() 
        loss_occ = nll_gaussian(keypoints_gt[:,:,0:2], keypoints_logits[:,:,0:2] , 10)
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


    def loss_kgnn3d(self, rt, valid, keypoint_kgnn2d, ratio_all):
        keypoint_kgnn2d = keypoint_kgnn2d.type(torch.FloatTensor).cuda() 
        shape_basis = rt[:,7:]
        #print(rt)
        shape = rt[:,7:]@(self.pca_component) + self.mean_shape
        shape = shape.view(-1,7,3)
        shape = torch.cat((shape,shape),2)
        shape[:,:,5]=-shape[:,:,5]
        shape = shape.view(-1,14,3)
        projected_points = orthographic_proj_withz(shape, rt[:,0:7], ratio_all)
        keypoint_kgnn2d = keypoint_kgnn2d*valid.unsqueeze(2).type(torch.FloatTensor).cuda()
        projected_points = projected_points*valid.unsqueeze(2).type(torch.FloatTensor).cuda()
        #print(projected_points)
        loss_kgnn3d = nll_gaussian(keypoint_kgnn2d[:,:,0:2], projected_points[:,:,0:2] , 10)
        if torch.isnan(loss_kgnn3d):
              asasas
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
