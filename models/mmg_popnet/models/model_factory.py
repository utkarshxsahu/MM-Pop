"""
Model factory
"""

from models.mlp_model import MLPModel
from models.graphsage_model import GraphSAGEModel
from models.gat_model import GATModel
from models.text_graphsage import TextGraphSAGE


def create_model(model_type, model_params, device):
    """
    Create a model instance
    
    Args:
        model_type: 'mlp', 'graphsage', 'gat', or 'text_graphsage'
        model_params: dict with model-specific parameters
        device: torch device
    
    Returns:
        model instance
    """
    if model_type == 'mlp':
        model = MLPModel(
            input_dim=model_params['input_dim'],
            output_dim=model_params['output_dim'],
            hidden_dims=model_params.get('hidden_dims', [128, 64]),
            dropout=model_params.get('dropout', 0.2)
        )
    
    elif model_type == 'graphsage':
        model = GraphSAGEModel(
            node_feat_dim=model_params['node_feat_dim'],
            global_feat_dim=model_params['global_feat_dim'],
            output_dim=model_params['output_dim'],
            num_layers=model_params.get('num_layers', 3),
            hidden_dim=model_params.get('hidden_dim', 64),
            aggregator_type=model_params.get('aggregator_type', 'mean'),
            dropout=model_params.get('dropout', 0.2),
            final_mlp_layers=model_params.get('final_mlp_layers', 2),
            final_mlp_hidden=model_params.get('final_mlp_hidden', 128),
            use_images=model_params.get('use_images', False),
            freeze_image_encoder=model_params.get('freeze_image_encoder', True),
            unfreeze_last_n_layers=model_params.get('unfreeze_last_n_layers', 0),
            use_per_target_mlp=model_params.get('use_per_target_mlp', False),
            image_projection_dim=model_params.get('image_projection_dim', 32),
            ablate_image=model_params.get('ablate_image', False),
            ablate_topology=model_params.get('ablate_topology', False)
        )

    elif model_type == 'text_graphsage':
        # Separate GNN params
        gnn_params = {
            'num_layers': model_params.get('num_layers', 3),
            'hidden_dim': model_params.get('hidden_dim', 128),
            'aggregator_type': model_params.get('aggregator_type', 'mean'),
            'dropout': model_params.get('dropout', 0.25),
            'final_mlp_layers': model_params.get('final_mlp_layers', 3),
            'final_mlp_hidden': model_params.get('final_mlp_hidden', 256),
            'use_per_target_mlp': model_params.get('use_per_target_mlp', False),
            'image_projection_dim': model_params.get('image_projection_dim', 32),
            'ablate_image': model_params.get('ablate_image', False),
            'ablate_topology': model_params.get('ablate_topology', False)
        }
        
        model = TextGraphSAGE(
            structural_feat_dim=model_params['structural_feat_dim'], 
            global_feat_dim=model_params['global_feat_dim'],
            output_dim=model_params['output_dim'],
            transformer_name=model_params.get('transformer_name', 'sentence-transformers/all-MiniLM-L6-v2'),
            gnn_params=gnn_params,
            text_projection_dim=model_params.get('text_projection_dim', 32),
            use_images=model_params.get('use_images', False),
            freeze_image_encoder=model_params.get('freeze_image_encoder', True),
            unfreeze_last_n_layers=model_params.get('unfreeze_last_n_layers', 0),
            ablate_text=model_params.get('ablate_text', False),
            ablate_image=model_params.get('ablate_image', False),
            ablate_topology=model_params.get('ablate_topology', False),
            ablate_temporal=model_params.get('ablate_temporal', False)
        )

    elif model_type == 'gat':  
        model = GATModel(
            node_feat_dim=model_params['node_feat_dim'],
            global_feat_dim=model_params['global_feat_dim'],
            output_dim=model_params['output_dim'],
            num_layers=model_params.get('num_layers', 2),
            hidden_dim=model_params.get('hidden_dim', 32),
            heads=model_params.get('heads', 4),
            dropout=model_params.get('dropout', 0.2),
            final_mlp_layers=model_params.get('final_mlp_layers', 2),
            final_mlp_hidden=model_params.get('final_mlp_hidden', 128),
            use_images=model_params.get('use_images', False),
            freeze_image_encoder=model_params.get('freeze_image_encoder', True),
            unfreeze_last_n_layers=model_params.get('unfreeze_last_n_layers', 0),
            image_projection_dim=model_params.get('image_projection_dim', 32)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)
