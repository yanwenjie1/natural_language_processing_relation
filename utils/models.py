# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/5
@Time    : 13:46
@File    : models.py
@Function: XX
@Other: XX
"""
import math

import torch
import torch.nn as nn
from utils.base_model import BaseModel


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids, entity_labels, relation_labels):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.entity_labels = entity_labels
        self.relation_labels = relation_labels


class GlobalPointer(nn.Module):
    """
    参考：https://kexue.fm/archives/8373
    """

    def __init__(self, hidden_size, heads, head_size, rope=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads  # 多头注意力
        self.head_size = head_size
        self.RoPE = rope
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, self.heads * self.head_size * 2, bias=use_bias)
        self.max_len = max_len
        if self.RoPE:
            position_embedding = self.sinusoidal_position_embedding(max_len, head_size)
            self.register_buffer('position_embedding', position_embedding)

    @staticmethod
    def sinusoidal_position_embedding(seq_len, output_dim):
        """
        旋转位置编码  https://kexue.fm/archives/8265
        :param seq_len:
        :param output_dim:
        :return:
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

    def forward(self, inputs, mask=None):
        """
        :param inputs: [batch_size, max_len, embeddings]
        :param mask: [batch_size, max_len], padding部分为0
        :return:
        """
        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]
        assert seq_len == self.max_len

        outputs = self.dense(inputs)  # outputs:(batch_size, seq_len, len(entity)*inner_dim*2)
        outputs = torch.split(outputs, self.head_size * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.head_size], outputs[..., self.head_size:]

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.position_embedding.repeat((batch_size, *([1] * len(self.position_embedding.shape))))
            pos_emb = torch.reshape(pos_emb, (batch_size, seq_len, self.head_size)).to(outputs.device)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        if mask is not None:
            pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.heads, seq_len, seq_len)
            logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits / self.head_size ** 0.5


def sinusoidal_position_embedding(seq_len, output_dim):
    """
    旋转位置编码  https://kexue.fm/archives/8265
    :param seq_len:
    :param output_dim:
    :return:
    """
    position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
    indices = torch.arange(0, output_dim // 2, dtype=torch.float)
    indices = torch.pow(10000, -2 * indices / output_dim)
    embeddings = position_ids * indices
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    return embeddings


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """

    def __init__(self, hidden_size, heads, head_size, rope=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = rope
        self.tril_mask = tril_mask

        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        ''' inputs: [batch_size, max_len, embeddings]
            mask: [batch_size, max_len], padding部分为0
        '''
        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]
        sequence_output = self.p_dense(inputs)
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]
        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size ** 0.5  # [btz, seq_len, seq_len]
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,
                                                                                          2)  # [btz, head_size, seq_len,2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2,
                                                                               3)  # [btz, head_size, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.heads, seq_len, seq_len)
            logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits.contiguous()


class GlobalPointerRe(BaseModel):
    def __init__(self, args):
        super(GlobalPointerRe, self).__init__(bert_dir=args.bert_dir,
                                              dropout_prob=args.dropout_prob,
                                              model_name=args.model_name)
        self.args = args
        self.entity_output = GlobalPointer(hidden_size=self.base_config.hidden_size,
                                           heads=self.args.num_entity_tags,
                                           max_len=self.args.max_seq_len,
                                           head_size=self.args.head_size)
        self.relation_output = GlobalPointer(hidden_size=self.base_config.hidden_size,
                                             heads=self.args.num_relation_tags * 2,
                                             head_size=self.args.head_size,
                                             max_len=self.args.max_seq_len,
                                             rope=False,
                                             tril_mask=False)

    def forward(self, token_ids, attention_masks, token_type_ids):
        output = self.bert_module(token_ids, attention_masks, token_type_ids)
        sequence_output = output[0]
        entity_output = self.entity_output(sequence_output, attention_masks)  # [btz, heads, seq_len, seq_len]
        relation_output = self.relation_output(sequence_output, attention_masks)  # [btz, heads, seq_len, seq_len]
        return entity_output, relation_output


class RoPEPositionEncoding(nn.Module):
    """
    旋转式位置编码: https://kexue.fm/archives/8265 适用于EfficientGlobalPointer
    """

    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)

    def forward(self, qw, seq_dim=-2):
        # [batch_size, max_len, heads, head_size] / [batch_size, max_len, head_size]
        # 默认最后两个维度为[seq_len, hdsz]? 不太对吧
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


def get_sinusoid_encoding_table(n_position, d_hid):
    '''Returns: [seq_len, d_hid]
    '''
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table
