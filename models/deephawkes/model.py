# deephawkes/model.py
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepHawkes(nn.Module):
    """
    Strict DeepHawkes:
      - user embedding lookup
      - GRU over each diffusion path
      - time-decay weighted pooling over paths (learned discrete bins)
      - MLP to outputs
    """
    def __init__(
        self,
        num_users: int,
        embedding_dim: int,
        hidden_dim: int,
        num_bins: int,
        out_dim: int,
        dropout: float = 0.1,
        pad_idx: int = 0,  # we use 0 as UNK; still safe for padding if needed
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.out_dim = int(out_dim)
        self.horizon_dim = 5

        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim, padding_idx=None)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Learnable discrete decay weights λ_l (one per time bin)
        # We apply softplus so weights are positive.
        self.lambda_raw = nn.Parameter(torch.zeros(self.num_bins))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + self.horizon_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, out_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch keys:
          - paths_padded: [P, Lmax]
          - lengths: [P]
          - bin_ids: [P]
          - tree_ptr: [P] values in [0..B-1]
        returns:
          - y_hat: [B, out_dim]
        """
        paths = batch["paths_padded"]      # [P, Lmax]
        lengths = batch["lengths"]         # [P]
        bin_ids = batch["bin_ids"]         # [P]
        tree_ptr = batch["tree_ptr"]       # [P]
        horizon_one_hot = batch["horizon_one_hot"]  # [B, 5]
        device = paths.device

        P, Lmax = paths.shape
        if P == 0:
            # shouldn't happen in valid batches
            return torch.zeros((0, self.out_dim), device=device)

        # Embed users
        x = self.user_emb(paths)  # [P, Lmax, emb]

        # Pack for GRU
        # sort by length for packing
        lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
        x_sorted = x.index_select(0, idx_sort)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        _, h_last = self.gru(packed)  # h_last: [1, P, hidden]
        path_vecs_sorted = h_last.squeeze(0)  # [P, hidden]

        # unsort back
        _, idx_unsort = torch.sort(idx_sort)
        path_vecs = path_vecs_sorted.index_select(0, idx_unsort)  # [P, hidden]

        # Time-decay weights
        lam = F.softplus(self.lambda_raw)  # [num_bins], positive
        w = lam.index_select(0, bin_ids.clamp(0, self.num_bins - 1))  # [P]
        w = w.unsqueeze(-1)  # [P, 1]

        weighted = w * path_vecs  # [P, hidden]

        # Pool per tree (sum)
        B = int(tree_ptr.max().item()) + 1
        c = torch.zeros((B, weighted.size(-1)), device=device)
        c.index_add_(0, tree_ptr, weighted)

        conditioned = torch.cat([c, horizon_one_hot], dim=-1)

        # Predict
        y_hat = self.mlp(conditioned)  # [B, out_dim]
        return y_hat
