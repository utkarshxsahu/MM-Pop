"""
GAT Model with a single MLP output head.

Architecture:
    node features → GATConv (up + down) → pool → SingleMLPHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from models.base_model import BaseModel
from models.task_heads import SingleMLPHead


class GATModel(BaseModel):
    def __init__(self, node_feat_dim, global_feat_dim, output_dim,
                 num_layers=3, hidden_dim=32, heads=4, dropout=0.2,
                 final_mlp_layers=2, final_mlp_hidden=128, use_images=False,
                 freeze_image_encoder=True, unfreeze_last_n_layers=0,
                 image_projection_dim=32):
        super().__init__(node_feat_dim, output_dim)

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.use_images = use_images
        self.freeze_image_encoder = freeze_image_encoder
        self.image_projection_dim = image_projection_dim

        self.gat_out_dim = hidden_dim * heads

        # Upstream Convs
        self.convs_up = nn.ModuleList()
        self.convs_up.append(GATConv(node_feat_dim, hidden_dim, heads=heads, concat=True))
        for _ in range(num_layers - 1):
            self.convs_up.append(GATConv(self.gat_out_dim, hidden_dim, heads=heads, concat=True))

        # Downstream Convs
        self.convs_down = nn.ModuleList()
        self.convs_down.append(GATConv(node_feat_dim, hidden_dim, heads=heads, concat=True))
        for _ in range(num_layers - 1):
            self.convs_down.append(GATConv(self.gat_out_dim, hidden_dim, heads=heads, concat=True))

        # Image embedding processing
        if use_images:
            if freeze_image_encoder:
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

        # ------------------------------------------------------------------ #
        #  Single MLP output head                                             #
        # ------------------------------------------------------------------ #
        trunk_input_dim = (self.gat_out_dim * 4) + global_feat_dim
        if use_images:
            trunk_input_dim += image_projection_dim

        self.output_head = SingleMLPHead(
            input_dim=trunk_input_dim,
            num_layers=final_mlp_layers,
            hidden_dim=final_mlp_hidden,
            dropout=dropout
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index = data.edge_index
        edge_index_rev = data.edge_index_rev

        # Upstream pass
        h_up = x
        for i, conv in enumerate(self.convs_up):
            h_up = conv(h_up, edge_index)
            h_up = F.elu(h_up)
            if i < self.num_layers - 1:
                h_up = self.dropout(h_up)

        # Downstream pass
        h_down = x
        for i, conv in enumerate(self.convs_down):
            h_down = conv(h_down, edge_index_rev)
            h_down = F.elu(h_down)
            if i < self.num_layers - 1:
                h_down = self.dropout(h_down)

        h_combined = torch.cat([h_up, h_down], dim=-1)
        root_emb = h_combined[data.root_mask]
        graph_emb = global_mean_pool(h_combined, batch)

        combined = torch.cat([root_emb, graph_emb, data.global_features], dim=-1)

        if self.use_images:
            if self.clip_model is not None:
                img_features = self.clip_model.get_image_features(
                    pixel_values=data.image_pixels.squeeze(1))
                projected = self.image_projection(img_features)
                projected = self.image_dropout(projected)
            else:
                projected = self.image_projection(data.image_embedding.squeeze(1))

            batch_size = data.image_embedding.size(0)
            no_img_batch = self.no_image_embedding.unsqueeze(0).expand(batch_size, -1)
            img_embs = torch.where(data.has_image.unsqueeze(1), projected, no_img_batch)
            combined = torch.cat([combined, img_embs], dim=-1)

        return self.output_head(combined)
