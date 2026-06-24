"""Standalone graph-aware selector for triple scoring.

This module fuses the embedding construction used by SubgraphRAG's retriever
with the CSM-style global selector architecture used by ``SecondStageSelector``.
The resulting ``Selector`` can be trained and used for inference without
instantiating or depending on ``Retriever``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class PEConv(MessagePassing):
    """Mean-aggregation propagation layer for positional encodings."""

    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class DDE(nn.Module):
    """Directional distance encoding over forward and reverse graph edges."""

    def __init__(self, num_rounds, num_reverse_rounds):
        super().__init__()
        self.layers = nn.ModuleList(PEConv() for _ in range(num_rounds))
        self.reverse_layers = nn.ModuleList(PEConv() for _ in range(num_reverse_rounds))

    def forward(self, topic_entity_one_hot, edge_index, reverse_edge_index):
        result_list = []

        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)

        return result_list


class GlobalLayer(nn.Module):
    """CSM-style global encoder over a query item and candidate triple items."""

    def __init__(self, num_layer, global_hidden_size, num_heads, dropout=0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            global_hidden_size,
            num_heads,
            batch_first=True,
            activation=F.gelu,
            norm_first=False,
            dropout=dropout,
        )
        self.context_attn = nn.TransformerEncoder(encoder_layer, num_layer)
        self.query_emb = nn.Embedding(2, global_hidden_size)
        self.query_norm = nn.LayerNorm(global_hidden_size)

    def forward(self, pair_feat, pair_nums):
        device = pair_feat.device
        if len(set(pair_nums)) == 1:
            group_size = pair_nums[0]
            batch_pair_feat = pair_feat.view(len(pair_nums), group_size, -1)
        else:
            batch_pair_feat = nn.utils.rnn.pad_sequence(
                pair_feat.split(pair_nums), batch_first=True
            )

        batch_size, max_len = batch_pair_feat.shape[:2]
        query_tags = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        query_tags[:, 0] = 1
        batch_pair_feat = self.query_norm(batch_pair_feat + self.query_emb(query_tags))

        encoded = self.context_attn(
            batch_pair_feat,
            src_key_padding_mask=self.attention_mask(pair_nums).to(device),
            mask=self.query_isolated_attention_mask(pair_nums).to(device),
        )
        return torch.cat([encoded[i, :n, :] for i, n in enumerate(pair_nums)], dim=0)

    @staticmethod
    def attention_mask(pair_nums):
        max_len = max(pair_nums)
        lengths = torch.tensor(pair_nums)
        return torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)

    @staticmethod
    def query_isolated_attention_mask(pair_nums):
        max_len = max(pair_nums)
        mask = torch.zeros(max_len, max_len, dtype=torch.bool)
        mask[0, 1:] = True
        return mask


class Selector(nn.Module):
    """Standalone selector that builds triple embeddings and scores triples."""

    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        global_hidden_size=256,
        num_heads=8,
        global_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.topic_pe = topic_pe
        self.global_hidden_size = global_hidden_size
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.dde = DDE(**DDE_kwargs)

        self.triple_feature_size = 4 * emb_size
        if topic_pe:
            self.triple_feature_size += 2 * 2
        self.triple_feature_size += 2 * 2 * (
            DDE_kwargs["num_rounds"] + DDE_kwargs["num_reverse_rounds"]
        )

        self.triple_feat_map = nn.Linear(self.triple_feature_size, global_hidden_size)
        self.query_feat_map = nn.Linear(emb_size, global_hidden_size)
        self.global_layer = GlobalLayer(
            num_layer=global_layers,
            global_hidden_size=global_hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.proj_head = nn.Sequential(
            nn.Linear(global_hidden_size * 2, global_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_size, global_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_size // 2, 1),
        )

    def build_triple_features(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
    ):
        device = entity_embs.device
        h_e = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(torch.LongTensor([0]).to(device)).expand(
                    num_non_text_entities, -1
                ),
            ],
            dim=0,
        )
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)
        h_e_list.extend(self.dde(topic_entity_one_hot, edge_index, reverse_edge_index))
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        h_r = relation_embs[r_id_tensor]
        return torch.cat(
            [h_q.expand(len(h_r), -1), h_e[h_id_tensor], h_r, h_e[t_id_tensor]], dim=1
        )

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
    ):
        if len(h_id_tensor) == 0:
            return entity_embs.new_empty((0, 1))
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)

        triple_features = self.build_triple_features(
            h_id_tensor,
            r_id_tensor,
            t_id_tensor,
            q_emb,
            entity_embs,
            num_non_text_entities,
            relation_embs,
            topic_entity_one_hot,
        )
        query_feat = self.query_feat_map(q_emb[:1])
        context_feat = self.triple_feat_map(triple_features)
        pair_feat = torch.cat([query_feat, context_feat], dim=0)
        encoded = self.global_layer(pair_feat, [pair_feat.size(0)])

        query_encoded = encoded[0]
        context_encoded = encoded[1:]
        query_repeated = query_encoded.unsqueeze(0).expand(context_encoded.size(0), -1)
        return self.proj_head(torch.cat([query_repeated, context_encoded], dim=-1))
