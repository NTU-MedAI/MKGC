import torch
from torch import nn
import torch.nn.functional as F
import math
import copy
import pickle
import csv
from encoder.data_utils import load_data_from_df
from encoder.readout import Set2Set
import pdb
from dgl.nn.pytorch import NNConv
from os import fchdir
import torch as th
from torch.nn import init
from dgl import function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.readout import sum_nodes, broadcast_nodes, softmax_nodes


class SGATT(nn.Module):
    def __init__(self, node_fts_1,edge_fts_1, message_size, message_passes, out_fts):
    # def __init__(self, message_passes):
        super(SGATT, self).__init__()
        self.node_fts_1 = node_fts_1
        self.edge_fts_1 = edge_fts_1
        self.message_size = message_size
        self.message_passes = message_passes
        self.out_fts = out_fts
        self.max_d = 50
        self.input_dim_drug = 23532
        self.n_layer = 2
        self.emb_size=384
        self.dropout_rate = 0
        self.n_heads = 1
        self.hid_dim = 128

        # encoder
        self.hidden_size = 384
        self.intermediate_size = 1536
        self.num_attention_heads = 8
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1

        # specialized embedding with positional one
        self.emb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.cross_att = CrossAttentionBlock(hid_dim=self.hid_dim, n_heads=self.n_heads, dropout=self.dropout_rate)

        # dencoder
        self.decoder_trans_mpnn_cat = nn.Sequential(
            nn.Linear(406, 64),
            nn.ReLU(True),

            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),

            # output layer
            nn.Linear(32, 1)
        )
        # self.decoder_trans_mpnn_sum = nn.Sequential(
        #     nn.Linear(203, 32),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(32),
        #     # output layer
        #     nn.Linear(32, 1)
        # )

        self.decoder_trans_mpnn_sum = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            # output layer
            nn.Linear(32, 1)
        )

        self.decoder_1 = nn.Sequential(
             nn.Linear(50*384, 512),
             nn.ReLU(True),
             nn.BatchNorm1d(512),

             nn.Linear(512, 128)
        )

        self.project_node_feats = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.gru = nn.GRU(64, 64)

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

        self.gnn_layer = gnn(
            in_feats=64,
            out_feats=64,
            attn_fc=attn_fc,
            edge_func1=edge_network1,
            edge_func2=edge_network2,
            aggregator_type='sum'
        )

        self.lstm = th.nn.LSTM(128, 64,3)

        loaded_dict = pickle.load(open('/home/ntu/PycharmProjects/Hao/SAGTT/encoder/frag_RotatE_128_64_emb.pkl', 'rb'))
        entity_emb, relation_emb = loaded_dict['entity_emb'], loaded_dict['relation_emb']
        atom_emb = torch.randn((118, 128))
        node_emb = torch.cat((atom_emb, entity_emb), 0)
        bond_emb = torch.randn((4, 64))
        edge_emb = torch.cat((bond_emb, relation_emb), 0)
        self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=False)
        self.edge_emb = nn.Embedding.from_pretrained(edge_emb, freeze=False)

    def aggregate_message_1(self, nodes, node_neighbours, edges, mask):

        raise NotImplementedError
    # inputs are "batches" of shape (maximum number of nodes in batch, number of features)
    def update_1(self, nodes, messages):
        raise NotImplementedError
    # inputs are "batches" of same shape as the nodes passed to update
    # node_mask is same shape as inputs and is 1 if elements corresponding exists, otherwise 0
    def readout_1(self, hidden_nodes, input_nodes, node_mask):
        raise NotImplementedError

    def readout(self,input_nodes, node_mask):
        raise NotImplementedError
    def final_layer(self,out):

        raise NotImplementedError

    def KMPNN(self, g,entity_emb, relation_emb):
        try:
            node_feats = self.node_emb(g.ndata['h'])
            edge_feats = self.edge_emb(g.edata['e'])
            node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
            hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)
            for _ in range(6):
                node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
                node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
                node_feats = node_feats.squeeze(0)
            return node_feats
        except:
            return None

    def Set_readout(self, graph, feat):
        try:
            with graph.local_scope():
                batch_size = graph.batch_size
                h = (feat.new_zeros((3, batch_size, 64)),
                     feat.new_zeros((3, batch_size, 64)))
                q_star = feat.new_zeros(batch_size, 128)
                for _ in range(6):
                    q, h = self.lstm(q_star.unsqueeze(0), h)
                    q = q.view(batch_size, 64)
                    e = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
                    graph.ndata['e'] = e
                    alpha = softmax_nodes(graph, 'e')
                    graph.ndata['r'] = feat * alpha
                    readout = sum_nodes(graph, 'r')
                    q_star = th.cat([q, readout], dim=-1)
                return q_star
        except:
            return None

    def forward(self, adj_1, nd_1, ed_1,de_1,mask_1,bg,entity_emb, relation_emb):

        #Graph encoder
        edge_batch_batch_indices_1, edge_batch_node_indices_1, edge_batch_neighbour_indices_1 = adj_1.nonzero().unbind(-1)
        node_batch_batch_indices_1, node_batch_node_indices_1 = adj_1.sum(-1).nonzero().unbind(-1)
        node_batch_adj_1 = adj_1[node_batch_batch_indices_1, node_batch_node_indices_1, :]
        node_batch_size_1 = node_batch_batch_indices_1.shape[0]
        node_degrees_1 = node_batch_adj_1.sum(-1).long()
        max_node_degree_1 = node_degrees_1.max()
        node_batch_node_neighbours_1 = torch.zeros(node_batch_size_1, max_node_degree_1, self.node_fts_1)
        node_batch_edges_1 = torch.zeros(node_batch_size_1, max_node_degree_1, self.edge_fts_1)
        node_batch_neighbour_neighbour_indices_1 = torch.cat([torch.arange(i) for i in node_degrees_1])
        edge_batch_node_batch_indices_1 = torch.cat(
            [i * torch.ones(degree) for i, degree in enumerate(node_degrees_1)]
        ).long()
        node_batch_node_neighbour_mask_1 = torch.zeros(node_batch_size_1, max_node_degree_1)


        node_batch_node_neighbour_mask_1[edge_batch_node_batch_indices_1, node_batch_neighbour_neighbour_indices_1] = 1
        node_batch_edges_1[edge_batch_node_batch_indices_1, node_batch_neighbour_neighbour_indices_1, :] = \
            ed_1[edge_batch_batch_indices_1, edge_batch_node_indices_1, edge_batch_neighbour_indices_1, :]
        hidden_nodes_1 = nd_1.clone()

        for i in range(self.message_passes):

            node_batch_nodes_1 = hidden_nodes_1[node_batch_batch_indices_1, node_batch_node_indices_1, :]
            node_batch_node_neighbours_1[edge_batch_node_batch_indices_1, node_batch_neighbour_neighbour_indices_1, :] = \
                hidden_nodes_1[edge_batch_batch_indices_1, edge_batch_neighbour_indices_1, :]
            messages_1 = self.aggregate_message_1(
                node_batch_nodes_1, node_batch_node_neighbours_1.clone(), node_batch_edges_1, node_batch_node_neighbour_mask_1
            )
            hidden_nodes_1[node_batch_batch_indices_1, node_batch_node_indices_1, :] = self.update_1(
                node_batch_nodes_1, messages_1)

        batch_size=nd_1.size(0)
        node_mask_1 = (adj_1.sum(-1) != 0)
        output_1 = self.readout_1(hidden_nodes_1, nd_1, node_mask_1)
        kg_batch = self.KMPNN(bg,entity_emb, relation_emb)
        kg_out = self.Set_readout(bg,kg_batch)
        #Sequence encoder
        ex_d_mask = de_1.unsqueeze(1).unsqueeze(2)
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0

        d_emb = self.emb(de_1)  # batch_size x seq_length x embed_size

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
        d1_trans_fts = d_encoded_layers.view(batch_size, -1)

        d1_trans_fts_layer1 = self.decoder_1(d1_trans_fts)
        # final_emb = g_out + d1_trans_fts_layer1
        final_emb = self.cross_att(output_1,kg_out,d1_trans_fts_layer1)

        #feature hybrid
        # d1_cat_fts=torch.cat((d1_trans_fts_layer1,output_1),dim=1)
        # final_fts_sum= d1_cat_fts
        # result = self.decoder_trans_mpnn_cat(final_fts_cat)
        # result=self.decoder_trans_mpnn_sum(final_fts_sum)####transformer输出
        # result = self.decoder_trans_mpnn_sum(d1_trans_fts_layer1)
        return final_emb


# help classes

class gnn(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 attn_fc,
                 edge_func1,
                 edge_func2,
                 aggregator_type='mean',
                 residual=False,
                 bias=True):
        super(gnn, self).__init__()
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


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        b=torch.LongTensor(1,2)

        input_ids=input_ids.type_as(b)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)#【1.。。50】

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)


        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # adj_matrix = adj / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
        # adj_matrix = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # if trainable_lambda:
        #     softmax_attention, softmax_distance, softmax_adjacency = lambdas.cuda()
        #     p_weighted = softmax_attention * p_attn + softmax_adjacency * p_adj
        # else:
        #     lambda_attention, lambda_distance, lambda_adjacency = lambdas
        #     p_weighted = lambda_attention * p_attn + lambda_adjacency * p_adj

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        """
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Query and Value should always come from the same source (Aiming to forcus on), Key comes from the other source
        Self-Att : Both three Query, Key, Value come form the same source (For refining purpose)
        """

        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2, 3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale.cpu()

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix


class CrossAttentionBlock(nn.Module):

    def __init__(self, hid_dim,n_heads,dropout):
        super(CrossAttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.hidden_size = hid_dim
        self.dropout = dropout
        self.att = AttentionBlock(hid_dim=self.hidden_size, n_heads=self.n_heads, dropout=self.dropout)

    def forward(self, graph_feature,kg, sequence_feature):
        if kg is not None:
            g_out = graph_feature+kg
            g_out = g_out + self.att(sequence_feature, g_out, g_out)
            output = self.att(g_out, g_out, g_out)

            return output
        else:
            graph_feature = graph_feature + self.att(sequence_feature, graph_feature, graph_feature)
            output = self.att(graph_feature, graph_feature, graph_feature)

            return output


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)  # +注意力
        attention_output = self.output(self_output, input_tensor)  # +残差
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)  # 给向量加了残差和注意力机制
        intermediate_output = self.intermediate(attention_output)  # 给向量拉长
        layer_output = self.output(intermediate_output, attention_output)  # 把向量带着残差压缩回去

        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states