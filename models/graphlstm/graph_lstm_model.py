"""
Graph-structured bidirectional LSTM for tree-level regression.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

import config


class GraphLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(input_dim, 5 * hidden_dim)
        self.U = nn.Linear(hidden_dim, 5 * hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 5 * hidden_dim, bias=False)

    def forward(self, x, h_temp, c_temp, h_hier, c_hier):
        gates = self.W(x) + self.U(h_temp) + self.V(h_hier)
        i, f, g, c_tilde, o = gates.chunk(5, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.sigmoid(g)
        c_tilde = torch.tanh(c_tilde)
        o = torch.sigmoid(o)
        c_t = f * c_temp + g * c_hier + i * c_tilde
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class GraphLSTMModel(nn.Module):
    def __init__(
        self,
        struct_dim: int,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        n_targets: int = 6,
        dropout: float = 0.3,
        mlp_hidden: int = 128,
        use_horizon_conditioning: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_horizon_conditioning = use_horizon_conditioning

        input_dim = struct_dim + embed_dim
        head_input_dim = 2 * hidden_dim + (len(config.HORIZON_TAGS) if use_horizon_conditioning else 0)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fwd_cell = GraphLSTMCell(input_dim, hidden_dim)
        self.bwd_cell = GraphLSTMCell(input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(head_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_targets),
        )
        self.dropout = nn.Dropout(dropout)

    def load_pretrained_embeddings(self, vocab_path, embeddings_path, vocab_map_path=None):
        if not embeddings_path or not os.path.exists(embeddings_path):
            return False

        external_vocab = None
        if vocab_map_path and os.path.exists(vocab_map_path):
            with open(vocab_map_path) as handle:
                external_vocab = json.load(handle)

        with open(vocab_path) as handle:
            local_vocab = json.load(handle)
        embedding_array = np.load(embeddings_path)

        if external_vocab is None:
            if embedding_array.shape != tuple(self.embedding.weight.data.shape):
                raise ValueError(
                    f"Pretrained embedding shape {embedding_array.shape} does not match "
                    f"model shape {tuple(self.embedding.weight.data.shape)}"
                )
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_array).float())
            return True

        if embedding_array.shape[0] != len(external_vocab):
            raise ValueError("Pretrained embedding matrix rows do not match external vocab size")

        loaded = 0
        for token, local_idx in local_vocab.items():
            external_idx = external_vocab.get(token)
            if external_idx is None:
                continue
            self.embedding.weight.data[local_idx] = torch.from_numpy(embedding_array[external_idx]).float()
            loaded += 1
        return loaded > 0

    def _encode_text(self, token_ids, token_lens):
        embeds = self.embedding(token_ids)
        seq_len = token_ids.size(1)
        mask = torch.arange(seq_len, device=token_ids.device).unsqueeze(0) < token_lens.unsqueeze(1)
        embeds = embeds * mask.unsqueeze(-1).float()
        return embeds.sum(1) / token_lens.unsqueeze(1).float().clamp(min=1)

    def _run_lstm_waves(self, cell, all_x, all_depth, all_sib_pos, all_temp_idx, all_hier_idx, reverse):
        max_sib = 10_000
        total_nodes = all_x.size(0)
        device = all_x.device

        h0 = torch.zeros(self.hidden_dim, device=device)
        c0 = torch.zeros(self.hidden_dim, device=device)

        wave_keys = all_depth * max_sib + all_sib_pos
        unique_waves = wave_keys.unique(sorted=True)
        if reverse:
            unique_waves = unique_waves.flip(0)

        h_store = [None] * total_nodes
        c_store = [None] * total_nodes

        for wave_key in unique_waves:
            wave_idx = (wave_keys == wave_key).nonzero(as_tuple=True)[0]
            x_wave = all_x[wave_idx]

            temp_ids = all_temp_idx[wave_idx]
            hier_ids = all_hier_idx[wave_idx]
            h_temp = torch.stack([h_store[i.item()] if i.item() >= 0 and h_store[i.item()] is not None else h0 for i in temp_ids])
            c_temp = torch.stack([c_store[i.item()] if i.item() >= 0 and c_store[i.item()] is not None else c0 for i in temp_ids])
            h_hier = torch.stack([h_store[i.item()] if i.item() >= 0 and h_store[i.item()] is not None else h0 for i in hier_ids])
            c_hier = torch.stack([c_store[i.item()] if i.item() >= 0 and c_store[i.item()] is not None else c0 for i in hier_ids])

            h_wave, c_wave = cell(x_wave, h_temp, c_temp, h_hier, c_hier)
            for idx, global_idx in enumerate(wave_idx):
                h_store[global_idx.item()] = h_wave[idx]
                c_store[global_idx.item()] = c_wave[idx]

        return torch.stack([h_store[i] if h_store[i] is not None else h0 for i in range(total_nodes)])

    def forward(self, batch):
        device = next(self.parameters()).device
        x_parts, depth_parts, sib_pos_parts = [], [], []
        parent_parts, sib_pred_parts, child_parts, sib_succ_parts = [], [], [], []
        offsets = [0]

        for tree in batch:
            struct = tree["struct_feats"].to(device)
            token_ids = tree["token_ids"].to(device)
            token_lens = tree["token_lens"].to(device)
            text = self._encode_text(token_ids, token_lens)
            x = self.dropout(torch.cat([struct, text], dim=-1))
            node_count = x.size(0)
            offset = offsets[-1]

            x_parts.append(x)
            depth_parts.append(tree["depth_arr"].to(device))
            sib_pos_parts.append(tree["sib_pos_arr"].to(device))

            def shift(indices):
                indices = indices.clone().to(device)
                indices[indices >= 0] += offset
                return indices

            parent_parts.append(shift(tree["parent_idx"]))
            sib_pred_parts.append(shift(tree["sib_pred_idx"]))
            child_parts.append(shift(tree["first_child_idx"]))
            sib_succ_parts.append(shift(tree["sib_succ_idx"]))
            offsets.append(offset + node_count)

        all_x = torch.cat(x_parts)
        all_depth = torch.cat(depth_parts)
        all_sib_pos = torch.cat(sib_pos_parts)
        all_parent = torch.cat(parent_parts)
        all_sib_pred = torch.cat(sib_pred_parts)
        all_child = torch.cat(child_parts)
        all_sib_succ = torch.cat(sib_succ_parts)

        h_fwd = self._run_lstm_waves(self.fwd_cell, all_x, all_depth, all_sib_pos, all_sib_pred, all_parent, False)
        h_bwd = self._run_lstm_waves(self.bwd_cell, all_x, all_depth, all_sib_pos, all_sib_succ, all_child, True)
        h_all = torch.cat([h_fwd, h_bwd], dim=-1)

        embeddings = torch.stack([h_all[offsets[i]:offsets[i + 1]].mean(dim=0) for i in range(len(batch))])
        if self.use_horizon_conditioning:
            horizon_vec = torch.stack([tree["horizon_vec"] for tree in batch]).to(device)
            embeddings = torch.cat([embeddings, horizon_vec], dim=-1)
        return self.mlp(embeddings)
