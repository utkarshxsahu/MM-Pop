import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCasModel(nn.Module):
    """DeepCas: Random walks + GRU + Attention with local node embeddings."""
    
    def __init__(self, vocab_size, global_feat_dim, output_dim,
                 embedding_dim=128, hidden_dim=128, 
                 K=100, T=10, num_size_bins=20,
                 dropout=0.3, final_mlp_hidden=128,
                 pretrained_weights=None):  # <--- NEW ARGUMENT HERE
        """
        Args:
            vocab_size: Size of node vocabulary (max_local_id, e.g., 500)
            global_feat_dim: Dimension of global features
            output_dim: Number of prediction targets
            embedding_dim: Node embedding dimension
            hidden_dim: GRU hidden dimension
            K: Number of sampled paths per tree
            T: Length of each path
            num_size_bins: Number of bins for tree size
            dropout: Dropout rate
            final_mlp_hidden: Hidden dimension of final MLP
            pretrained_weights: Numpy array of shape (vocab_size, embedding_dim)
        """
        super().__init__()
        
        self.K = K
        self.T = T
        self.hidden_dim = hidden_dim
        
        # Node embeddings
        self.node_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # === CHANGE START: Load Pretrained Weights ===
        if pretrained_weights is not None:
            print(f"Loading pre-trained Node2Vec weights (Shape: {pretrained_weights.shape})...")
            # Convert numpy to torch tensor
            weights_tensor = torch.FloatTensor(pretrained_weights)
            
            # Verify dimensions match
            if weights_tensor.shape != (vocab_size, embedding_dim):
                raise ValueError(f"Shape mismatch! Model expects ({vocab_size}, {embedding_dim}), "
                                 f"but loaded ({weights_tensor.shape}).")
                                 
            # Copy into the embedding layer
            self.node_embedding.weight.data.copy_(weights_tensor)
            
            # Keep pretrained embeddings fixed to match the old setup.
            self.node_embedding.weight.requires_grad = False
            print("  -> Embeddings frozen.")
        else:
            print("WARNING: Using random initialization (Xavier Uniform)!")
            nn.init.xavier_uniform_(self.node_embedding.weight)
        # === CHANGE END ===

        print(f"Node embeddings: {vocab_size} vocab × {embedding_dim} dim = {vocab_size * embedding_dim:,} params")
        
        # Bi-directional GRU
        self.gru_forward = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.gru_backward = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # Attention over positions (multinomial distribution)
        self.position_attention = nn.Parameter(torch.randn(T))
        
        # Attention over sequences (geometric distribution, size-conditioned)
        self.sequence_attention = nn.Parameter(torch.randn(num_size_bins))
        
        # Final MLP
        mlp_input_dim = 2 * hidden_dim + global_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, final_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_mlp_hidden, final_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_mlp_hidden, output_dim)
        )
    
    def forward(self, paths_batch, global_features, tree_sizes, mini_batch_size=5):
        """
        Forward pass.
        """
        batch_size = paths_batch.size(0)
        device = paths_batch.device
        
        # 1. Embed all nodes in all paths
        path_embeddings = self.node_embedding(paths_batch)  # (batch, K, T, embedding_dim)
        
        # 2. Apply bi-directional GRU to each path
        flat_paths = path_embeddings.view(batch_size * self.K, self.T, -1)
        
        # Forward GRU
        gru_out_fwd, _ = self.gru_forward(flat_paths)
        # Backward GRU
        flat_paths_rev = torch.flip(flat_paths, dims=[1])
        gru_out_bwd, _ = self.gru_backward(flat_paths_rev)
        gru_out_bwd = torch.flip(gru_out_bwd, dims=[1])
        
        # Concatenate
        gru_out = torch.cat([gru_out_fwd, gru_out_bwd], dim=-1)
        gru_out = gru_out.view(batch_size, self.K, self.T, 2 * self.hidden_dim)
        
        # 3. Position attention (λ_i)
        lambda_weights = F.softmax(self.position_attention, dim=0)
        lambda_weights = lambda_weights.view(1, 1, self.T, 1)
        path_representations = (gru_out * lambda_weights).sum(dim=2)  # (batch, K, 2*hidden)
        
        # 4. Sequence attention (geometric distribution) - VECTORIZED VERSION
        size_bins = torch.floor(torch.log2(tree_sizes.float() + 1)).long()
        size_bins = torch.clamp(size_bins, 0, len(self.sequence_attention) - 1)
        p_geo = torch.sigmoid(self.sequence_attention[size_bins])  # (batch_size,)
        
        num_mini_batches = self.K // mini_batch_size
        
        # Reshape paths: (batch, K, 2H) → (batch, num_mb, mb_size, 2H)
        # This groups consecutive K paths into mini_batches
        path_representations_reshaped = path_representations.view(
            batch_size, num_mini_batches, mini_batch_size, 2 * self.hidden_dim
        )
        
        # Average within each mini-batch
        # (batch, num_mb, mb_size, 2H) → (batch, num_mb, 2H)
        mini_batch_reps = path_representations_reshaped.mean(dim=2)
        
        # Compute geometric weights: p_geo * (1 - p_geo)^i for i = 0, 1, ..., num_mb-1
        indices = torch.arange(num_mini_batches, device=device, dtype=torch.float32)  # (num_mb,)
        indices = indices.unsqueeze(0)  # (1, num_mb) for broadcasting
        
        p_geo_expanded = p_geo.unsqueeze(1)  # (batch, 1) for broadcasting
        
        # Geometric distribution for each batch item
        # (batch, 1) * (1, num_mb) → (batch, num_mb)
        geo_weights = p_geo_expanded * torch.pow(1 - p_geo_expanded, indices)
        
        # Normalize so each row sums to 1
        geo_weights = geo_weights / (geo_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Weighted sum over mini-batches
        # (batch, num_mb, 1) * (batch, num_mb, 2H) → (batch, num_mb, 2H)
        # Sum over dim=1 (mini-batches) → (batch, 2H)
        graph_representations = (geo_weights.unsqueeze(2) * mini_batch_reps).sum(dim=1)
        
        # 5. Concatenate with global features
        combined = torch.cat([graph_representations, global_features], dim=-1)
        
        # 6. Final MLP
        return self.mlp(combined)
    
    def get_num_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
