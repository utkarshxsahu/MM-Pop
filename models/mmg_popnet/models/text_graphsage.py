"""
TextGraphSAGE — End-to-End Transformer + GraphSAGE + Images.

Location: models/text_graphsage.py

Device strategy:
  DDP mode         : Each rank owns one full GPU. Transformer and GNN both
                     live on cfg.DEVICE (= cuda:{local_rank}). Activation
                     memory per rank is 1/world_size. This is the primary mode.

  Single process,
  2+ GPUs          : Model parallelism — transformer on cuda:0, GNN on cuda:1.
                     Activated only when DDP is NOT running.

  Single GPU       : Everything on cfg.DEVICE.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoModel
from models.graphsage_model import GraphSAGEModel
from models.base_model import BaseModel
import config.config as cfg


class TextGraphSAGE(BaseModel):

    def __init__(self, structural_feat_dim, global_feat_dim, output_dim,
                 transformer_name='sentence-transformers/all-MiniLM-L6-v2',
                 gnn_params=None, text_projection_dim=32, use_images=False,
                 freeze_image_encoder=True, unfreeze_last_n_layers=0,
                 ablate_text=False, ablate_image=False,
                 ablate_topology=False, ablate_temporal=False):
        super().__init__(structural_feat_dim, output_dim)

        self.use_images           = use_images
        self.freeze_image_encoder = freeze_image_encoder
        self.text_projection_dim  = text_projection_dim
        self.ablate_text          = ablate_text
        self.ablate_image         = ablate_image
        self.ablate_topology      = ablate_topology
        self.ablate_temporal      = ablate_temporal

        # Detect mode at construction time
        ddp_active     = dist.is_available() and dist.is_initialized()
        n_gpus         = cfg._N_GPUS
        # Use model parallelism only when running single-process with multiple GPUs
        self.use_model_par = (not ddp_active) and (n_gpus >= 2)

        if ddp_active:
            # Each rank has its own GPU — everything on cfg.DEVICE
            self.dev_transformer = cfg.DEVICE
            self.dev_gnn         = cfg.DEVICE
            mode = f"DDP single-rank ({cfg.DEVICE})"
        elif n_gpus >= 2:
            self.dev_transformer = torch.device('cuda:0')
            self.dev_gnn         = torch.device('cuda:1')
            mode = "model parallelism (transformer=cuda:0, GNN=cuda:1)"
        else:
            self.dev_transformer = cfg.DEVICE
            self.dev_gnn         = cfg.DEVICE
            mode = f"single GPU ({cfg.DEVICE})"

        print(f"[TextGraphSAGE] {mode}")

        # ------------------------------------------------------------------
        # 1. Transformer backbone
        # ------------------------------------------------------------------
        if ablate_text:
            print("[TextGraphSAGE] Text ablation active: transformer disabled")
            self.transformer = None
            transformer_out_dim = None
        else:
            print(f"Loading transformer: {transformer_name}")
            self.transformer = AutoModel.from_pretrained(transformer_name)
            transformer_out_dim = self.transformer.config.hidden_size

        #self.transformer.gradient_checkpointing_enable()

        if ablate_text:
            self.text_projection = None
        else:
            self.text_projection = nn.Sequential(
                nn.Linear(transformer_out_dim, 64, bias=True),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, text_projection_dim, bias=True),
            )
            for module in self.text_projection:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        # ------------------------------------------------------------------
        # 2. GNN head
        # ------------------------------------------------------------------
        effective_structural_dim = 0 if ablate_temporal else structural_feat_dim
        total_node_dim = text_projection_dim + effective_structural_dim

        if gnn_params is None:
            gnn_params = {}
        gnn_params['use_images']             = use_images
        gnn_params['freeze_image_encoder']   = freeze_image_encoder
        gnn_params['unfreeze_last_n_layers'] = unfreeze_last_n_layers
        gnn_params['ablate_image']           = ablate_image
        gnn_params['ablate_topology']        = ablate_topology

        print(f"GNN input_dim={total_node_dim} "
              f"({text_projection_dim} text + "
              f"{effective_structural_dim} structural)")

        self.gnn = GraphSAGEModel(
            node_feat_dim=total_node_dim,
            global_feat_dim=global_feat_dim,
            output_dim=output_dim,
            **gnn_params
        )

    def to(self, *args, **kwargs):
        if self.use_model_par:
            # Place sub-modules on their designated devices;
            # do NOT call super().to() which would move everything to one device
            if self.transformer is not None:
                self.transformer = self.transformer.to(self.dev_transformer)
            if self.text_projection is not None:
                self.text_projection = self.text_projection.to(self.dev_transformer)
            self.gnn             = self.gnn.to(self.dev_gnn)
            return self
        else:
            # DDP or single GPU — standard behaviour moves everything to cfg.DEVICE
            return super().to(*args, **kwargs)

    def forward(self, data):
        # ------------------------------------------------------------------
        # 1. Transformer pass
        # ------------------------------------------------------------------
        if self.ablate_text:
            text_projected = torch.zeros(
                data.x.size(0), self.text_projection_dim,
                dtype=data.x.dtype, device=self.dev_gnn
            )
        else:
            input_ids      = data.input_ids.to(self.dev_transformer)
            attention_mask = data.attention_mask.to(self.dev_transformer)

            outputs          = self.transformer(input_ids=input_ids,
                                                attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            mask_expanded    = attention_mask.unsqueeze(-1).expand(
                                   token_embeddings.size()).float()
            sum_embeddings   = torch.sum(token_embeddings * mask_expanded, 1)
            sum_mask         = torch.clamp(mask_expanded.sum(1), min=1e-9)
            text_embeddings  = sum_embeddings / sum_mask

            # Project down
            text_projected = self.text_projection(text_embeddings)

        # ------------------------------------------------------------------
        # 2. Move to GNN device and concatenate with structural features
        #    In DDP / single-GPU: dev_transformer == dev_gnn — no-op transfer.
        #    In model-parallel:   moves projected text from cuda:0 to cuda:1.
        # ------------------------------------------------------------------
        text_on_gnn  = text_projected.to(self.dev_gnn)
        if self.ablate_temporal:
            combined_x = text_on_gnn
        else:
            structural = data.x.to(self.dev_gnn)
            combined_x = torch.cat([text_on_gnn, structural], dim=1)

        original_x               = data.x
        data.x                   = combined_x
        data.edge_index          = data.edge_index.to(self.dev_gnn)
        data.edge_index_rev      = data.edge_index_rev.to(self.dev_gnn)
        data.batch               = data.batch.to(self.dev_gnn)
        data.global_features     = data.global_features.to(self.dev_gnn)
        data.root_mask           = data.root_mask.to(self.dev_gnn)
        if data.image_embedding is not None:
            data.image_embedding = data.image_embedding.to(self.dev_gnn)
        if data.image_pixels is not None:
            data.image_pixels    = data.image_pixels.to(self.dev_gnn)
        if data.has_image is not None:
            data.has_image       = data.has_image.to(self.dev_gnn)

        # ------------------------------------------------------------------
        # 3. GNN pass
        # ------------------------------------------------------------------
        output = self.gnn(data)
        data.x = original_x
        return output
