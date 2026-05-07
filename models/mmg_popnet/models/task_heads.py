"""
Task-grouped MLP output heads for multi-task prediction.

Architecture:
    shared_representation → {structural_head, participation_head, engagement_head}

The shared trunk (existing MLP body in each model) feeds into separate
lightweight heads, one per semantic group of targets. This allows each head
to learn group-specific feature weightings while the shared encoder captures
common propagation signals.

Why separate heads (for the paper):
  - structural targets are emergent whole-tree properties
  - participation measures distinct-user engagement (hybrid of structure + content)
  - engagement (root_score) is a root-post-level property that precedes thread growth
  These three groups have different predictive signals; forcing a shared output
  layer conflates them. Separate heads are standard in multi-task learning
  (Caruana 1997) and add negligible training overhead vs. the GNN/transformer.
"""

import torch
import torch.nn as nn
import config.feature_config as feat_cfg


class SingleMLPHead(nn.Module):
    """
    Shared MLP output head with configurable total depth.

    Replaces the grouped TaskHeads design. Using a single shared output projection
    allows the model to learn cross-target correlations in the final layer, and is
    compatible with the horizon-masking loss (where root_score is masked for
    non-final horizons while all other targets are still supervised).

    Args:
        input_dim  : dimensionality of the shared trunk output (z_T)
        hidden_dim : hidden layer width
        num_layers : total number of Linear layers in the head, including the
                     final output projection. By project convention, the old
                     baseline `final_mlp_layers=2` means:
                     Linear(in→hidden) → ReLU → Dropout → Linear(hidden→n_targets).
        dropout    : dropout probability after each hidden layer
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        if num_layers < 2:
            raise ValueError("SingleMLPHead requires num_layers >= 2")

        n_targets = len(feat_cfg.OUTPUT_TARGETS)
        layers = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, n_targets))
        self.mlp = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return len(feat_cfg.OUTPUT_TARGETS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch_size, input_dim)
        Returns:
            (batch_size, n_targets)
        """
        return self.mlp(x)


class TaskHeads(nn.Module):
    """
    Separate linear output head per TARGET_GROUP.

    Each head is a single linear layer applied to the shared trunk output.
    The shared trunk (the main MLP body) already provides a rich representation;
    per-group heads only need to project to their respective output dimensions.

    Output columns are concatenated in TARGET_GROUPS order, which matches
    OUTPUT_TARGETS order — this keeps compatibility with evaluator / loss code
    that indexes targets by position.

    Args:
        input_dim : dimensionality of the shared trunk output
        dropout   : dropout probability applied before each head (consistent
                    with the shared trunk's final dropout)
    """

    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()

        self.group_names = list(feat_cfg.TARGET_GROUPS.keys())
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict({
            group_name: nn.Linear(input_dim, len(targets))
            for group_name, targets in feat_cfg.TARGET_GROUPS.items()
        })

    @property
    def output_dim(self) -> int:
        return sum(len(v) for v in feat_cfg.TARGET_GROUPS.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch_size, input_dim) shared representation from trunk
        Returns:
            (batch_size, total_n_targets) concatenated per-group predictions,
            ordered to match OUTPUT_TARGETS
        """
        x = self.dropout(x)
        return torch.cat(
            [self.heads[name](x) for name in self.group_names],
            dim=1
        )


class PerTargetMLPHeads(nn.Module):
    """
    One independent MLP per target in OUTPUT_TARGETS.

    Each MLP takes the raw concatenated graph representation z_T directly
    (before any shared trunk compression) and produces a scalar prediction
    for its assigned target. This eliminates the representational bottleneck
    where a single shared trunk must simultaneously compress z_T into a space
    that serves all targets — each MLP instead learns its own compression
    tailored to its specific prediction problem.

    Architecture per target:
        z_T (trunk_input_dim)
          → Linear(trunk_input_dim → mlp_hidden) → ReLU → Dropout
          → [repeated for num_layers - 1]
          → Linear(mlp_hidden → 1)

    Output columns are concatenated in OUTPUT_TARGETS order, matching
    the cache/evaluator convention.

    Args:
        trunk_input_dim : dimensionality of z_T (the concatenated graph vector)
        num_layers      : number of linear layers in each per-target MLP
        mlp_hidden      : hidden dimension of each per-target MLP
        dropout         : dropout probability applied after each hidden layer
    """

    def __init__(self, trunk_input_dim: int, num_layers: int = 2,
                 mlp_hidden: int = 128, dropout: float = 0.2):
        super().__init__()

        self.target_names = list(feat_cfg.OUTPUT_TARGETS)
        self.dropout_p    = dropout

        def _build_mlp():
            layers  = []
            in_dim  = trunk_input_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(in_dim, mlp_hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = mlp_hidden
            layers.append(nn.Linear(mlp_hidden, 1))
            return nn.Sequential(*layers)

        self.mlps = nn.ModuleDict({
            target: _build_mlp()
            for target in self.target_names
        })

    @property
    def output_dim(self) -> int:
        return len(self.target_names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch_size, trunk_input_dim)  raw z_T vector
        Returns:
            (batch_size, n_targets)  predictions in OUTPUT_TARGETS order
        """
        return torch.cat(
            [self.mlps[t](x) for t in self.target_names],
            dim=1
        )
