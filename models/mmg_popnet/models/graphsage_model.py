"""
GraphSAGE model with a single MLP output head.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from models.base_model import BaseModel
from models.task_heads import SingleMLPHead, PerTargetMLPHeads


class GraphSAGEModel(BaseModel):
    def __init__(self, node_feat_dim, global_feat_dim, output_dim,
                 num_layers=3, hidden_dim=128, aggregator_type='mean', dropout=0.2,
                 final_mlp_layers=2, final_mlp_hidden=128, use_images=False,
                 freeze_image_encoder=True, unfreeze_last_n_layers=0,
                 use_per_target_mlp=False, image_projection_dim=32,
                 ablate_image=False, ablate_topology=False):
        super().__init__(node_feat_dim, output_dim)

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.use_images = use_images
        self.freeze_image_encoder = freeze_image_encoder
        self.use_per_target_mlp = use_per_target_mlp
        self.image_projection_dim = image_projection_dim
        self.ablate_image = ablate_image
        self.ablate_topology = ablate_topology

        if ablate_topology:
            # Node-wise replacement for bidirectional GraphSAGE: same depth and
            # hidden size, but no edge-dependent message passing.
            self.convs_up = None
            self.convs_down = None
            self.node_mlps_up = nn.ModuleList()
            self.node_mlps_down = nn.ModuleList()
            self.node_mlps_up.append(nn.Linear(node_feat_dim, hidden_dim))
            self.node_mlps_down.append(nn.Linear(node_feat_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.node_mlps_up.append(nn.Linear(hidden_dim, hidden_dim))
                self.node_mlps_down.append(nn.Linear(hidden_dim, hidden_dim))
        else:
            self.node_mlps_up = None
            self.node_mlps_down = None

            # Upstream Convs (Child -> Parent)
            self.convs_up = nn.ModuleList()
            self.convs_up.append(SAGEConv(node_feat_dim, hidden_dim, aggr=aggregator_type))
            for _ in range(num_layers - 1):
                self.convs_up.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator_type))

            # Downstream Convs (Parent -> Child)
            self.convs_down = nn.ModuleList()
            self.convs_down.append(SAGEConv(node_feat_dim, hidden_dim, aggr=aggregator_type))
            for _ in range(num_layers - 1):
                self.convs_down.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator_type))

        # Image embedding processing
        if use_images:
            if ablate_image:
                self.no_image_embedding = nn.Parameter(torch.randn(image_projection_dim) * 0.01)
                self.image_projection = None
                self.clip_model = None
            elif freeze_image_encoder:
                self.image_projection = nn.Linear(512, image_projection_dim, bias=False)
                nn.init.xavier_uniform_(self.image_projection.weight)
                self.no_image_embedding = nn.Parameter(torch.randn(image_projection_dim) * 0.01)
                self.clip_model = None
            else:
                from transformers import CLIPModel
                import config.config as cfg

                print(f"Loading CLIP Model (full) for unfrozen training...")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

                for param in self.clip_model.parameters():
                    param.requires_grad = False

                if unfreeze_last_n_layers > 0:
                    total_layers = len(self.clip_model.vision_model.encoder.layers)
                    print(f"  Unfreezing last {unfreeze_last_n_layers} of {total_layers} CLIP vision layers")
                    for layer_idx in range(total_layers - unfreeze_last_n_layers, total_layers):
                        for param in self.clip_model.vision_model.encoder.layers[layer_idx].parameters():
                            param.requires_grad = True

                self.image_projection = nn.Linear(512, image_projection_dim, bias=False)
                nn.init.xavier_uniform_(self.image_projection.weight)
                self.no_image_embedding = nn.Parameter(torch.randn(image_projection_dim) * 0.01)
                self.image_dropout = nn.Dropout(cfg.IMAGE_DROPOUT)

        # Output head: SingleMLPHead or per-target MLPs (controlled by use_per_target_mlp).
        trunk_input_dim = (hidden_dim * 4) + node_feat_dim + global_feat_dim
        if use_images:
            trunk_input_dim += image_projection_dim

        if use_per_target_mlp:
            # Each target gets its own full MLP starting from z_T directly.
            self.mlp_trunk   = None
            self.output_head = PerTargetMLPHeads(
                trunk_input_dim=trunk_input_dim,
                num_layers=final_mlp_layers,
                mlp_hidden=final_mlp_hidden,
                dropout=dropout
            )
        else:
            # Shared head baseline was 2 layers total with hidden width 128:
            # Linear(in→hidden) → ReLU → Dropout → Linear(hidden→n_targets).
            # Active experiment uses a deeper/wider shared head via
            # final_mlp_layers/final_mlp_hidden.
            self.mlp_trunk   = None
            self.output_head = SingleMLPHead(
                input_dim=trunk_input_dim,
                num_layers=final_mlp_layers,
                hidden_dim=final_mlp_hidden,
                dropout=dropout
            )

    def forward(self, data):
        x, batch = data.x, data.batch

        if self.ablate_topology:
            h_up = x
            for i, layer in enumerate(self.node_mlps_up):
                h_up = torch.relu(layer(h_up))
                if i < self.num_layers - 1:
                    h_up = self.dropout(h_up)

            h_down = x
            for i, layer in enumerate(self.node_mlps_down):
                h_down = torch.relu(layer(h_down))
                if i < self.num_layers - 1:
                    h_down = self.dropout(h_down)
        else:
            edge_index = data.edge_index
            edge_index_rev = data.edge_index_rev

            # Upstream pass
            h_up = x
            for i, conv in enumerate(self.convs_up):
                h_up = conv(h_up, edge_index)
                h_up = torch.relu(h_up)
                if i < self.num_layers - 1:
                    h_up = self.dropout(h_up)

            # Downstream pass
            h_down = x
            for i, conv in enumerate(self.convs_down):
                h_down = conv(h_down, edge_index_rev)
                h_down = torch.relu(h_down)
                if i < self.num_layers - 1:
                    h_down = self.dropout(h_down)

        h_combined = torch.cat([h_up, h_down], dim=-1)
        root_emb = h_combined[data.root_mask]
        root_raw = x[data.root_mask]
        graph_emb = global_mean_pool(h_combined, batch)

        combined = torch.cat([root_emb, root_raw, graph_emb, data.global_features], dim=-1)

        if self.use_images:
            batch_size   = data.has_image.size(0)

            # Default: learnable no-image embedding for every graph in the batch
            img_embs = self.no_image_embedding.unsqueeze(0).expand(batch_size, -1).clone()

            if not self.ablate_image:
                has_img_mask = data.has_image.view(-1).bool()   # (batch_size,)
                # Only run CLIP / projection on the graphs that actually have an image
                if has_img_mask.any():
                    if self.clip_model is not None:
                        img_features = self.clip_model.get_image_features(
                            pixel_values=data.image_pixels[has_img_mask].squeeze(1))
                        projected = self.image_projection(img_features)
                        projected = self.image_dropout(projected)
                    else:
                        projected = self.image_projection(
                            data.image_embedding[has_img_mask].squeeze(1))
                    img_embs[has_img_mask] = projected.to(dtype=img_embs.dtype)

            combined = torch.cat([combined, img_embs], dim=-1)

        return self.output_head(combined)
