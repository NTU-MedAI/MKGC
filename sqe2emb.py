import numpy as np
import pandas as pd
from subword_nmt.apply_bpe import BPE
import codecs
import torch
from torch import nn

def sqe2emb_encoder(x):
    ## Sequence encoder parameter
    vocab_path = './ESPF/drug_codes_chembl.txt'
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
    max_d = 50
    t1 = dbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])
    except:
        i1 = np.array([0])
        print('error:', x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    word_embeddings = nn.Embedding(23552, 384)
    position_embeddings = nn.Embedding(50, 384)
    gamma = nn.Parameter(torch.ones(384))
    beta = nn.Parameter(torch.zeros(384))
    dropout = nn.Dropout(0.1)
    linear1 = nn.Linear(19200, 128)

    de_1 = torch.from_numpy(i)
    ex_d_mask = de_1.unsqueeze(1).unsqueeze(2)
    ex_d_mask = (1.0 - ex_d_mask) * -10000.0

    b = torch.LongTensor(1, 2)
    input_ids = de_1.type_as(b)
    input_ids = input_ids.reshape(input_ids.shape[0], 1)
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # 【1.。。50】
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    words_embeddings = word_embeddings(input_ids)
    position_embeddings = position_embeddings(position_ids)
    embeddings = words_embeddings + position_embeddings
    u = embeddings.mean(-1, keepdim=True)
    s = (embeddings - u).pow(2).mean(-1, keepdim=True)
    x = (embeddings - u) / torch.sqrt(s + 384)
    embeddings = gamma * x + beta
    embeddings = dropout(embeddings)
    embeddings = embeddings.reshape(1, 19200)
    embeddings = linear1(embeddings)
    return embeddings


class AMDE(nn.Module):

    def __init__(self):
    # def __init__(self, message_passes):
        super(AMDE, self).__init__()
        self.max_d = 50
        self.input_dim_drug = 23532
        self.n_layer = 2
        self.emb_size=384
        self.dropout_rate=0

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

    def forward(self, de_1):
        ex_d_mask = de_1.unsqueeze(1).unsqueeze(2)
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0

        d_emb = self.emb(de_1)  # batch_size x seq_length x embed_size

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
        d1_trans_fts = d_encoded_layers.view(batch_size, -1)

        d1_trans_fts_layer1 = self.decoder_1(d1_trans_fts)
        final_emb = d1_trans_fts_layer1


        return final_emb


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
