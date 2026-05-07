"""
Dataset Configuration

Window 0 = root-only condition (no cascade, just root post).
Add 0 to any dataset's windows list to enable root-only for that dataset.

root_only_reference_window: required when windows contains 0.
  Specifies which cascade window to use for loading posts/meta/splits
  when running root-only. This window does NOT need to appear in windows[]
  — it is used silently just to get the data. If windows[] also contains
  non-zero entries, the first non-zero entry takes priority over this field.

Examples:
  'windows': [0, 5, 15, 30]   → root-only + all cascade windows
  'windows': [0, 20, 50, 90]  → root-only + all cascade windows
  'windows': [5, 15, 30]      → cascade only (no root-only)
  'windows': [0]              → root-only only (uses root_only_reference_window)
"""
import os

# Root directories
BLUESKY_ROOT = "/scratch/dgl/Social_Network/bluesky/processed"
REDDIT_ROOT = "/scratch/dgl/Social_Network/reddit/filtered/tree_full"
REDDIT_EMBEDDING_ROOT = "/scratch/dgl/Social_Network/reddit/embeddings/minilm384_fp16"

DATASETS = {
    'bluesky': {
        'type': 'bluesky',
        'paths': {
            'thread_metadata': os.path.join(BLUESKY_ROOT, "discussion_trees/samples/tree_65k/thread_metadata_updated_added.parquet"),
            'thread_posts': os.path.join(BLUESKY_ROOT, "discussion_trees/samples/tree_65k/thread_posts_with_all_labels2.parquet"),
            'user_degrees': os.path.join(BLUESKY_ROOT, "followers/user_degrees.csv"),
            'early_window_dir': os.path.join(BLUESKY_ROOT, "discussion_trees/samples/tree_65k/gnn_snapshots"),
            'embedding_dir': os.path.join(BLUESKY_ROOT, "embeddings/minilm384_fp16"),
            'output_dir': os.path.join(BLUESKY_ROOT, "discussion_trees/samples/tree_65k/gnn_results/Early_Window"),
            'image_dir': None
        },
        'windows': [0, 2, 10],          # add 0 for root-only: [0, 2, 10,20]
        'root_only_reference_window': 2,  # used when windows=[0] only
        'embedding_mode': 'sharded',
        'has_user_degrees': True,
        'has_images': False
    },
    'gaming': {
        'type': 'reddit',
        'paths': {
            'thread_metadata': os.path.join(REDDIT_ROOT, "reddit_gaming_metadata.parquet"),
            'thread_posts': os.path.join(REDDIT_ROOT, "reddit_gaming_posts.parquet"),
            'user_degrees': None,
            'early_window_dir': os.path.join(REDDIT_ROOT, "snapshots/gaming"),
            'embedding_dir': os.path.join(REDDIT_EMBEDDING_ROOT, "gaming"),
            'output_dir': os.path.join(REDDIT_ROOT, "gnn_results/gaming"),
            'image_dir': '/scratch/dgl/Social_Network/reddit/image_embeddings/gaming'
        },
        'windows': [0, 20, 50],          # add 0 for root-only: [0, 20, 50, 90]
        'root_only_reference_window': 20, # used when windows=[0] only
        'embedding_mode': 'single_file',
        'has_user_degrees': False,
        'has_images': True
    },
    'futurology': {
        'type': 'reddit',
        'paths': {
            'thread_metadata': os.path.join(REDDIT_ROOT, "reddit_futurology_metadata.parquet"),
            'thread_posts': os.path.join(REDDIT_ROOT, "reddit_futurology_posts.parquet"),
            'user_degrees': None,
            'early_window_dir': os.path.join(REDDIT_ROOT, "snapshots/futurology"),
            'embedding_dir': os.path.join(REDDIT_EMBEDDING_ROOT, "futurology"),
            'output_dir': os.path.join(REDDIT_ROOT, "gnn_results/futurology"),
            'image_dir': '/scratch/dgl/Social_Network/reddit/image_embeddings/futurology'
        },
        'windows': [90],         # add 0 for root-only: [0, 30, 90, 180]
        'root_only_reference_window': 30, # used when windows=[0] only
        'embedding_mode': 'single_file',
        'has_user_degrees': False,
        'has_images': True
    },
    'ama': {
        'type': 'reddit',
        'paths': {
            'thread_metadata': os.path.join(REDDIT_ROOT, "reddit_ama_metadata.parquet"),
            'thread_posts': os.path.join(REDDIT_ROOT, "reddit_ama_posts.parquet"),
            'user_degrees': None,
            'early_window_dir': os.path.join(REDDIT_ROOT, "snapshots/ama"),
            'embedding_dir': os.path.join(REDDIT_EMBEDDING_ROOT, "ama"),
            'output_dir': os.path.join(REDDIT_ROOT, "gnn_results/ama"),
            'image_dir': None
        },
        'windows': [0, 15, 30],          # add 0 for root-only: [0, 15, 30, 60]
        'root_only_reference_window': 15, # used when windows=[0] only
        'embedding_mode': 'single_file',
        'has_user_degrees': False,
        'has_images': False
    }
}