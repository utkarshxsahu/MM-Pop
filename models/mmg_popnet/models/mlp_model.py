"""
MLP model with a single MLP output head.

Architecture:
    root + aggregated + global features → shared MLP trunk → SingleMLPHead
"""

import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.task_heads import SingleMLPHead


class MLPModel(BaseModel):
    """
    MLP for predicting thread metrics from root + aggregated + global features.
    Uses a single shared MLP output head.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64], dropout=0.2):
        super().__init__(input_dim, output_dim)

        # Shared trunk — all hidden layers except the last
        trunk_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.mlp_trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()

        # Single MLP output head using the last hidden_dim
        final_hidden = hidden_dims[-1] if hidden_dims else input_dim
        self.output_head = SingleMLPHead(
            input_dim=prev_dim,
            hidden_dim=final_hidden,
            dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, n_targets) — ordered per OUTPUT_TARGETS
        """
        shared = self.mlp_trunk(x)
        return self.output_head(shared)