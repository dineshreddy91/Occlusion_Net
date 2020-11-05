import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
#from utils import my_softmax, get_offdiag_indices, gumbel_softmax

import pdb

_EPS = 1e-10


class Flatten(nn.Module):
    def forward(self, input):
        """
        Returns the forward input of the input.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        return input.view(input.size(0), -1)
    
class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        """
        Initialize the gradient.

        Args:
            self: (todo): write your description
            n_in: (int): write your description
            n_hid: (int): write your description
            n_out: (str): write your description
            do_prob: (int): write your description
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights. weights.

        Args:
            self: (todo): write your description
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        """
        Batch normalization.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            n_in: (int): write your description
            n_hid: (int): write your description
            n_out: (str): write your description
            do_prob: (int): write your description
        """
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights.

        Args:
            self: (todo): write your description
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob
    

class GraphEncoder3D(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_kps, do_prob=0., factor=True):
        """
        Initialize the hps

        Args:
            self: (todo): write your description
            n_in: (int): write your description
            n_hid: (int): write your description
            n_out: (str): write your description
            n_kps: (int): write your description
            do_prob: (int): write your description
            factor: (float): write your description
        """
        super(GraphEncoder3D, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP graph encoder.")
        self.fc_out = nn.Linear(n_hid*n_kps, n_out)
        self.flatten   = Flatten()
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights.

        Args:
            self: (todo): write your description
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        """
        Return the edges of * x

        Args:
            self: (todo): write your description
            x: (todo): write your description
            rel_rec: (todo): write your description
            rel_send: (todo): write your description
        """
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """
        Return the edge from * rel_records.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            rel_rec: (todo): write your description
            rel_send: (todo): write your description
        """
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        """
        Forward forward

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            rel_rec: (todo): write your description
            rel_send: (todo): write your description
        """
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.flatten(x) 
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.flatten(x) 

        return self.fc_out(x)    



class GraphEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        """
        Initialize h_weights

        Args:
            self: (todo): write your description
            n_in: (int): write your description
            n_hid: (int): write your description
            n_out: (str): write your description
            do_prob: (int): write your description
            factor: (float): write your description
        """
        super(GraphEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP graph encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights.

        Args:
            self: (todo): write your description
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        """
        Return the edges of * x

        Args:
            self: (todo): write your description
            x: (todo): write your description
            rel_rec: (todo): write your description
            rel_send: (todo): write your description
        """
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        """
        Return the edge from * rel_records.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            rel_rec: (todo): write your description
            rel_send: (todo): write your description
        """
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        """
        Parameters : meth.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            rel_rec: (todo): write your description
            rel_send: (todo): write your description
        """
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)    

class GraphDecoder(nn.Module):

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            n_in_node: (int): write your description
            edge_types: (str): write your description
            msg_hid: (int): write your description
            msg_out: (str): write your description
            n_hid: (int): write your description
            do_prob: (int): write your description
            skip_first: (str): write your description
        """
        super(GraphDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned graph decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):
        """
        Parameters ---------- single step.

        Args:
            self: (todo): write your description
            single_timestep_inputs: (bool): write your description
            rel_rec: (str): write your description
            rel_send: (todo): write your description
            single_timestep_rel_type: (todo): write your description
        """

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)
#        print(pred.shape,single_timestep_inputs.shape)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
            rel_type: (str): write your description
            rel_rec: (todo): write your description
            rel_send: (todo): write your description
            pred_steps: (todo): write your description
        """
        # NOTE: Assumes that we have the same graph across all samples.


        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, :, :]
        #asa
        curr_rel_type = rel_type[:, :, :]
        preds=[]
        #print(curr_rel_type.shape)
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        #for step in range(0, pred_steps):
        last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
        preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1),
                 preds[0].size(2)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, :, :] = preds[i]

        pred_all = output[:, :, :]

        # NOTE: We potentially over-predicted (stored in future_pred). Unused.
        # future_pred = output[:, (inputs.size(1) - 1):, :, :]

        return pred_all#.transpose(1, 2).contiguous()    
    

