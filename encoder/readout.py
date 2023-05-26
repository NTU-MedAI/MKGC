import dgl
import torch
from dgl.nn.pytorch import WeightAndSum
import torch as th
import torch.nn as nn
import numpy as np
from dgl.readout import sum_nodes, broadcast_nodes, softmax_nodes

class Set2Set(nn.Module):
    def __init__(self):
        super(Set2Set, self).__init__()
        self.input_dim = 64
        self.output_dim = 2 * 64
        self.n_iters = 6
        self.n_layers = 3
        self.lstm = th.nn.LSTM(128, 64,3)
        self.reset_parameters()
        self.out_dim = 128

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                 feat.new_zeros((self.n_layers, batch_size, self.input_dim)))

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
                graph.ndata['e'] = e
                alpha = softmax_nodes(graph, 'e')
                graph.ndata['r'] = feat * alpha
                readout = sum_nodes(graph, 'r')
                q_star = th.cat([q, readout], dim=-1)
                print(q_star)
            return q_star