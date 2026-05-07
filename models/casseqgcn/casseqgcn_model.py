"""
CasSeqGCN model with horizon conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import config


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, hidden, laplacian):
        return F.relu(self.linear(torch.bmm(laplacian, hidden)))


class DynamicRouting(nn.Module):
    def __init__(self, in_dim, out_dim, n_iter=config.N_ROUTING_ITER):
        super().__init__()
        self.n_iter = n_iter
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, hidden, node_mask):
        projected = self.proj(hidden)
        batch_size, n_nodes, _ = projected.shape
        logits = torch.zeros(batch_size, n_nodes, device=hidden.device)
        logits = logits.masked_fill(~node_mask, float("-inf"))

        pooled = None
        for iteration in range(self.n_iter):
            coeff = F.softmax(logits, dim=1).unsqueeze(-1)
            pooled = (coeff * projected).sum(dim=1)
            if iteration < self.n_iter - 1:
                similarity = (
                    F.normalize(projected, dim=-1)
                    * F.normalize(pooled, dim=-1).unsqueeze(1)
                ).sum(dim=-1)
                logits = similarity.masked_fill(~node_mask, float("-inf"))

        return pooled


class CasSeqGCN(nn.Module):
    """GCN -> routing -> LSTM -> horizon-conditioned MLP."""

    def __init__(self, model_cfg, n_targets=len(config.TARGET_NAMES)):
        super().__init__()
        gcn_dims = [config.NODE_FEATURE_DIM] + [model_cfg["gcn_hidden"]] * model_cfg["gcn_layers"]
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(gcn_dims[idx], gcn_dims[idx + 1]) for idx in range(model_cfg["gcn_layers"])]
        )
        self.routing = DynamicRouting(
            model_cfg["gcn_hidden"],
            config.SNAPSHOT_EMBED_DIM,
            n_iter=config.N_ROUTING_ITER,
        )
        self.lstm = nn.LSTM(
            input_size=config.SNAPSHOT_EMBED_DIM,
            hidden_size=model_cfg["lstm_hidden"],
            num_layers=model_cfg["lstm_layers"],
            batch_first=True,
            dropout=model_cfg["dropout"] if model_cfg["lstm_layers"] > 1 else 0.0,
        )
        self.dropout = nn.Dropout(model_cfg["dropout"])
        mlp_input_dim = model_cfg["lstm_hidden"] + len(config.HORIZON_ORDER)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, model_cfg["lstm_hidden"]),
            nn.ReLU(),
            nn.Dropout(model_cfg["dropout"]),
            nn.Linear(model_cfg["lstm_hidden"], n_targets),
        )

    def forward(self, snapshots, laplacian, node_mask, snap_mask, horizon_onehot):
        _, n_snapshots, _, _ = snapshots.shape
        snapshot_vecs = []

        for snap_idx in range(n_snapshots):
            hidden = snapshots[:, snap_idx, :, :]
            for layer in self.gcn_layers:
                hidden = layer(hidden, laplacian)
            snapshot_vecs.append(self.routing(hidden, node_mask))

        sequence = torch.stack(snapshot_vecs, dim=1)
        sequence = sequence * snap_mask.unsqueeze(-1).float()

        seq_lens = snap_mask.sum(dim=1).cpu().clamp(min=1)
        packed = pack_padded_sequence(sequence, seq_lens, batch_first=True, enforce_sorted=False)
        _, (hidden_state, _) = self.lstm(packed)

        final_hidden = self.dropout(hidden_state[-1])
        conditioned = torch.cat([final_hidden, horizon_onehot], dim=-1)
        return self.mlp(conditioned)
