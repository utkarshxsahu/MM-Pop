"""
Model architecture configurations
"""

MODEL_CONFIGS = {
    'mlp': {
        'type': 'mlp',
        'hyperparams': {
            'n_layers': {'type': 'int', 'low': 2, 'high': 4},
            'hidden_dim_0': {'type': 'int', 'low': 64, 'high': 256},
            'hidden_dim_1': {'type': 'int', 'low': 64, 'high': 256},
            'hidden_dim_2': {'type': 'int', 'low': 64, 'high': 256},
            'hidden_dim_3': {'type': 'int', 'low': 64, 'high': 256},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [256, 512, 1024]},
        }
    },
    'graphsage': {
        'type': 'graphsage',
        'hyperparams': {
            'num_layers': {'type': 'int', 'low': 2, 'high': 6},
            'hidden_dim': {'type': 'int', 'low': 32, 'high': 128},
            'aggregator_type': {'type': 'categorical', 'choices': ['mean', 'max', 'lstm']},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'final_mlp_layers': {'type': 'int', 'low': 2, 'high': 4},
            'final_mlp_hidden': {'type': 'int', 'low': 64, 'high': 256},
            'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [256, 512, 1024]},
        }
    },

    'text_graphsage': {
        'type': 'text_graphsage',
        'hyperparams': {
            'num_layers': {'type': 'int', 'low': 2, 'high': 4},
            'hidden_dim': {'type': 'int', 'low': 64, 'high': 256},
            'aggregator_type': {'type': 'categorical', 'choices': ['mean', 'max']},
            'batch_size': {'type': 'categorical', 'choices': [32, 64]} # Lower due to Transformer
        }
    },

    'gat': {
        'type': 'gat',
        'hyperparams': {
            'num_layers': {'type': 'int', 'low': 2, 'high': 4},
            'hidden_dim': {'type': 'int', 'low': 16, 'high': 64},  # Smaller because of heads
            'heads': {'type': 'categorical', 'choices': [2, 4, 8]},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.6}, # GAT often needs higher dropout
            'final_mlp_layers': {'type': 'int', 'low': 2, 'high': 4},
            'final_mlp_hidden': {'type': 'int', 'low': 64, 'high': 256},
            'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [256, 512]}
        }
    }
}
