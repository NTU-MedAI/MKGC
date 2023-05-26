import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from dgl.nn.pytorch import NNConv
"""Torch Module for NNConv layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
from os import fchdir
import torch as th
from torch.nn import init

from dgl import function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import numpy as np
import time


class KMPNN(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 attn_fc,
                 edge_func1,
                 edge_func2,
                 aggregator_type='mean',
                 residual=False,
                 bias=True):
        super(KMPNN, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.attn_fc = attn_fc
        self.edge_func1 = edge_func1
        self.edge_func2 = edge_func2
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = init.calculate_gain('relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'attn_e': F.leaky_relu(a)}

    def message_func1(self, edges):
        return {'m1': edges.src['h'] * edges.data['w1'], 'attn_e1': edges.data['attn_e'], 'z1': edges.src['z']}

    def message_func2(self, edges):
        return {'m2': edges.src['h'] * edges.data['w2'], 'attn_e2': edges.data['attn_e'], 'z2': edges.src['z']}

    def reduce_func1(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn_e1'], dim=1).unsqueeze(-1)
        h = th.sum(alpha * nodes.mailbox['m1'], dim=1)
        return {'neigh1': h}

    def reduce_func2(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn_e2'], dim=1).unsqueeze(-1)
        h = th.sum(alpha * nodes.mailbox['m2'], dim=1)
        return {'neigh2': h}

    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # (n, d_in, 1)
            graph.srcdata['h'] = feat_src.unsqueeze(-1)

            # (n, d_in, d_out)
            graph.edata['w1'] = self.edge_func1(efeat).view(-1, self._in_src_feats, self._out_feats)
            graph.edata['w2'] = self.edge_func2(efeat).view(-1, self._in_src_feats, self._out_feats)

            graph.ndata['z'] = feat_src
            graph.apply_edges(self.edge_attention)
            # pdb.set_trace()
            # (n, d_in, d_out)
            edges1 = th.nonzero(graph.edata['etype'] == 0).squeeze(1).int()  # bonds
            edges2 = th.nonzero(graph.edata['etype'] == 1).squeeze(1).int()  # rels

            # graph.send_and_recv(edges1, fn.u_mul_e('h', 'w1', 'm'), self.reducer('m', 'neigh1'))
            graph.send_and_recv(edges1, self.message_func1, self.reduce_func1)
            graph.send_and_recv(edges2, self.message_func2, self.reduce_func2)
            rst1 = graph.dstdata['neigh1'].sum(dim=1)
            rst2 = graph.dstdata['neigh2'].sum(dim=1)  # (n, d_out)
            rst = rst1 + rst2  # (n, d_out)

            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst

class KMPNNGNN(nn.Module):
    def __init__(self, args, entity_emb, relation_emb):
        super(KMPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.num_step_message_passing = 6
        attn_fc = nn.Linear(2 * 64, 1, bias=False)
        edge_network1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64)
        )
        edge_network2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64)
        )

        self.gnn_layer = KMPNN(
            in_feats=64,
            out_feats=64,
            attn_fc=attn_fc,
            edge_func1=edge_network1,
            edge_func2=edge_network2,
            aggregator_type='sum'
        )

        self.gru = nn.GRU(64, 64)
        self.out_dim = 64

        # self.node_emb = nn.Embedding(343, args['node_indim'])
        # self.edge_emb = nn.Embedding(21, args['edge_indim'])

        atom_emb = torch.randn((118, 128))
        node_emb = torch.cat((atom_emb, entity_emb),0)
        bond_emb = torch.randn((4,64))
        edge_emb = torch.cat((bond_emb, relation_emb),0)
        self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=False)
        self.edge_emb = nn.Embedding.from_pretrained(edge_emb, freeze=False)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g):
        node_feats = self.node_emb(g.ndata['h'])
        edge_feats = self.edge_emb(g.edata['e'])
        print('node_feats',node_feats)
        print('edge_feats',edge_feats)
        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats