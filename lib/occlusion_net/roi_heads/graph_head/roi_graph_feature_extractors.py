import torch
from torch import nn
from torch.nn import functional as F
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable

from ... import registry
from maskrcnn_benchmark.modeling.poolers import Pooler

from maskrcnn_benchmark.layers import Conv2d
from .modules import GraphEncoder,GraphDecoder, GraphEncoder3D
from .utils import * 
from sklearn.decomposition import PCA

def graph_to_heatmap(graph_features):	
        heatmap = torch.zeros([count,num_keypoints,width,width], dtype=torch.float64).cuda()
        index_y = graph_features[:,:,0].type(torch.int32)
        index_x = graph_features[:,:,1].type(torch.int32)
        value = graph_features[:,:,2]
        loop1 = 0
        for ind_1 in index_x:
            loop2 = 0
            for ind_2 in ind_1:
               heatmap[loop1,loop2,ind_2,index_y[loop1,loop2]] = value[loop1,loop2]
            loop2 = loop2+1
        loop1 = loop1+1 

def heatmaps_to_graph(heatmaps):

      count = heatmaps.shape[0]
      num_keypoints = heatmaps.shape[1]
      width = heatmaps.shape[2]

      heatmaps = heatmaps.view(count,num_keypoints,width*width)

      values ,index= torch.max(heatmaps,2)
      index_y = index.div(width)
      index_x = index - index_y.mul(width)
      index_y = index_y.type(torch.cuda.FloatTensor).view(count,num_keypoints,1)
      index_x = index_x.type(torch.cuda.FloatTensor).view(count,num_keypoints,1)
      index_y = (index_y - (width/2))/(width/2)
      index_x = (index_x - (width/2))/(width/2)
      values = values.type(torch.cuda.FloatTensor).view(count,num_keypoints,1)

      graph_features = torch.cat((index_x,index_y,values),2)
      return graph_features

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

def pca_computation(path):
        pca_3d = np.load(path)
        #first_half = pca_3d[:,0:8,:]
        #second_half = pca_3d[:,8:,:]
        #sample_rows = np.zeros((pca_3d.shape[0],1,pca_3d.shape[2]))
        #pca_3d = np.concatenate((first_half,sample_rows), axis=1)
        #pca_3d = np.concatenate((pca_3d,second_half), axis=1)
        #pca_3d = np.concatenate((pca_3d,sample_rows), axis=1)
        Xtrain = np.zeros((pca_3d.shape[0],18))
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

@registry.ROI_GRAPH_FEATURE_EXTRACTORS.register("graphRCNNFeatureExtractor")
class graphRCNNFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super(graphRCNNFeatureExtractor, self).__init__()
        self.encoder = GraphEncoder(cfg.MODEL.ROI_GRAPH_HEAD.DIMS,
                                    cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_HIDDEN,
                                    cfg.MODEL.ROI_GRAPH_HEAD.EDGE_TYPES,
                                    cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_DROPOUT,
                                    cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_FACTOR)
        self.decoder = GraphDecoder(n_in_node=cfg.MODEL.ROI_GRAPH_HEAD.DIMS,
                                 edge_types=cfg.MODEL.ROI_GRAPH_HEAD.EDGE_TYPES,
                                 msg_hid=cfg.MODEL.ROI_GRAPH_HEAD.DECODER_HIDDEN,
                                 msg_out=cfg.MODEL.ROI_GRAPH_HEAD.DECODER_HIDDEN,
                                 n_hid=cfg.MODEL.ROI_GRAPH_HEAD.DECODER_HIDDEN,
                                 do_prob=cfg.MODEL.ROI_GRAPH_HEAD.DECODER_DROPOUT,
                                 skip_first=cfg.MODEL.ROI_GRAPH_HEAD.SKIP_FIRST)
        self.encoder_rt = GraphEncoder3D(cfg.MODEL.ROI_GRAPH_HEAD.DIMS,
                                    cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_HIDDEN,
                                    cfg.MODEL.ROI_GRAPH_HEAD.PARAMS_3D,
                                    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES,
                                    cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_DROPOUT,
                                    cfg.MODEL.ROI_GRAPH_HEAD.ENCODER_FACTOR)
        # Generate off-diagonal interaction graph
        self.off_diag = np.ones([cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES, cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES]) - np.eye(cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES)

        self.rel_rec = np.array(encode_onehot(np.where(self.off_diag)[1]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(self.off_diag)[0]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(self.rel_rec)
        self.rel_send = torch.FloatTensor(self.rel_send)

        self.encoder= self.encoder.cuda()
        self.encoder_rt = self.encoder_rt.cuda()
        self.decoder = self.decoder.cuda()
        self.rel_rec = self.rel_rec.cuda()
        self.rel_send = self.rel_send.cuda()

        self.mean_shape, self.pca_component = pca_computation('data/pca_3d_cad.npy')



       
    def forward(self, x, ratio):
        logits = self.encoder(x,self.rel_rec,self.rel_send)
        edges = my_softmax(logits, -1)
        KGNN2D = self.decoder(x, edges, self.rel_rec, self.rel_send,0)# args.prediction_steps)
        rt = self.encoder_rt(x, self.rel_rec, self.rel_send)
        shape_basis = rt[:,7:]
        shape = self.mean_shape# + shape_basis@(self.pca_component)
        shape = shape.view(-1,6,3)
        shape = torch.cat((shape,shape),2)
        shape[:,:,5]=-shape[:,:,5]
        shape = shape.view(-1,12,3)
        idx = 8
        b = torch.zeros(shape.shape[0],1, 3)
        b = b.cuda()
        shape = torch.cat([shape[:,:idx], b, shape[:,idx:]], 1) # considering the exhaust
        shape = torch.cat([shape, b], 1)# considering the middle of the boundingbox
        projected_points = orthographic_proj_withz(shape, rt[:,0:7], ratio)

        return logits,KGNN2D,projected_points


def make_roi_graph_feature_extractor(cfg):
    func = registry.ROI_GRAPH_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_GRAPH_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)
