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

def graph_to_heatmap(graph_features):	
        heatmap = torch.zeros([count,num_keypoints,width,width], dtype=torch.float64).cuda()
        index_x = graph_features[:,:,0].type(torch.int32)
        index_y = graph_features[:,:,1].type(torch.int32)
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
      index_x = index.div(width)
      index_y = index - index_x.mul(width)
      index_x = index_x.type(torch.cuda.FloatTensor).view(count,num_keypoints,1)
      index_y = index_y.type(torch.cuda.FloatTensor).view(count,num_keypoints,1)
      values = values.type(torch.cuda.FloatTensor).view(count,num_keypoints,1)

      graph_features = torch.cat((index_x,index_y,values),2)
      return graph_features


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




       
    def forward(self, x):
        logits = self.encoder(x,self.rel_rec,self.rel_send)
        edges = my_softmax(logits, -1)
        #print(x)
        KGNN2D = self.decoder(x, edges, self.rel_rec, self.rel_send,0)# args.prediction_steps)
        #print(KGNN2D)
        KGNN3D = self.encoder_rt(x, self.rel_rec, self.rel_send)
        return logits,KGNN2D,KGNN3D


def make_roi_graph_feature_extractor(cfg):
    func = registry.ROI_GRAPH_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_GRAPH_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg)
