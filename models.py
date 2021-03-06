"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import BertTokenizer, BertModel


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


#  Pass in 512 as the max_seq_length in run_squad.py
BERT_OUT_SIZE = 768
QUESTION_LEN = 60
CONTEXT_LEN = 449
TOTAL_SEQ_LEN = QUESTION_LEN + CONTEXT_LEN + 3 # To account for 2*[SEP] and [CLS] tokens

class Tevon(nn.Module):
    def __init__(self, h1=128, h2=128, drop_prob=0.2):
        super(Tevon, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.compress_features = nn.Linear(BERT_OUT_SIZE, h1)
        self.drop1 = nn.Dropout(drop_prob)
        self.norm1 = nn.LayerNorm(h1)
        self.compress_time = nn.Linear(TOTAL_SEQ_LEN, h2)
        self.drop2 = nn.Dropout(drop_prob)
        self.norm2 = nn.LayerNorm(h2)
        self.linear3 = nn.Linear(h1*h2, 2 * BERT_OUT_SIZE)
        self.norm3 = nn.LayerNorm(BERT_OUT_SIZE)

    def forward(self, input_ids, segment_ids, mask, start_positions=None, end_positions=None):
        bert_encodings, _ = self.bert(input_ids, segment_ids, mask, output_all_encoded_layers=False)  # (batch, time, embedding_size)
        x = torch.relu(self.compress_features(bert_encodings))  # First we compress the feature length to h1. output shape = (batch, T, h1)
        x = self.drop1(x)
        x = self.norm1(x)
        x = torch.relu(self.compress_time(x.permute(0, 2, 1)))  # We compress the time dimension. output shape = (batch, h1, h2)
        x = self.drop2(x)
        x = self.norm2(x)
        x = self.linear3(x.view(x.size(0), -1))  # We concatenate the last two dimensions. output shape = (batch, 2 * BERT_OUT_SIZE)
        start, end = x.split(BERT_OUT_SIZE, dim=-1)  # (batch x embedding_size)
        start = self.norm3(start)
        end = self.norm3(end)
        start_logits = torch.bmm(bert_encodings, start.unsqueeze(-1)).squeeze(-1)
        end_logits = torch.bmm(bert_encodings, end.unsqueeze(-1)).squeeze(-1)

        # We compute the loss inside the forward pass in order to take full advantage of Multi-GPU training.
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class TevonCNN(nn.Module):
    def __init__(self, h1=512, h2=256, h3=128, drop_prob=0.2):
        super(TevonCNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        # PERMUTE HERE
        # when padding = (kernel_size - 1) / 2 (stride=dilation=1), the time dimension remains constant
        self.conv1 = nn.Conv1d(BERT_OUT_SIZE, h1, kernel_size=6, stride=2, padding=2)
        self.norm1 = nn.LayerNorm(TOTAL_SEQ_LEN // 2)
        self.conv2 = nn.Conv1d(h1, h2, kernel_size=6, stride=2, padding=2)
        self.norm2 = nn.LayerNorm(TOTAL_SEQ_LEN // 4)
        self.conv3 = nn.Conv1d(h2, h3, kernel_size=6, stride=2, padding=2)
        self.norm3 = nn.LayerNorm(TOTAL_SEQ_LEN // 8)
        # REVERT PERMUTE
        # UPSAMPLE TIME DIMS
        # This choice of params keeps the embedding dim constant
        self.convT1 = nn.ConvTranspose1d(TOTAL_SEQ_LEN // 8, TOTAL_SEQ_LEN // 2, kernel_size=5, padding=2)
        self.norm4 = nn.LayerNorm(h3)
        self.convT2 = nn.ConvTranspose1d(TOTAL_SEQ_LEN // 2, TOTAL_SEQ_LEN, kernel_size=5, padding=2)
        self.norm5 = nn.LayerNorm(h3)
        self.linear = nn.Linear(h3, 2)

    def forward(self, input_ids, segment_ids, mask, start_positions=None, end_positions=None):
        # bert_encodings.shape = (batch, time, embedding_size)
        bert_encodings, _ = self.bert(input_ids, segment_ids, mask, output_all_encoded_layers=False)
        x = bert_encodings.permute(0, 2, 1)  # We permute in order for channels dim to match embedding dim.
        # x.shape = (batch, embedding_size, time)
        x = torch.relu(self.conv1(x))  # out shape = (batch, time / 2, h1)
        x = self.norm1(x)
        x = torch.relu(self.conv2(x)) # out shape = (batch, time / 4, h2)
        x = self.norm2(x)
        x = torch.relu(self.conv3(x)) # out shape = (batch, time / 8, h3)
        x = self.norm3(x)
        # UPSAMPLING TIME DIM
        x = x.permute(0, 2, 1)
        x = torch.relu(self.convT1(x))
        x = self.norm4(x)
        x = torch.relu(self.convT2(x))
        x = self.norm5(x)
        x = self.linear(x)  # We concatenate the last two dimensions. out shape = (batch, time, 2)
        start_logits, end_logits = x.split(1, dim=-1)  # start and end shape = (batch, time)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # We compute the loss inside the forward pass in order to take full advantage of Multi-GPU training.
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits