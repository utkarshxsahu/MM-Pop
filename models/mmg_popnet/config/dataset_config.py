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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "mmg_popnet")
BLUESKY_ROOT = os.path.join(DATASETS_ROOT, "bluesky")
REDDIT_ROOT = os.path.join(DATASETS_ROOT, "reddit")

DATASETS = {
    'bluesky': {
        'type': 'bluesky',
        'paths': {
            'thread_metadata': os.path.join(BLUESKY_ROOT, "metadata/thread_metadata_updated_added.parquet"),
            'thread_posts': os.path.join(BLUESKY_ROOT, "metadata/thread_posts_with_all_labels2.parquet"),
            'user_degrees': os.path.join(BLUESKY_ROOT, "metadata/user_degrees.csv"),
            'early_window_dir': os.path.join(BLUESKY_ROOT, "snapshots"),
            'embedding_dir': os.path.join(BLUESKY_ROOT, "embeddings"),
            'output_dir': RESULTS_ROOT,
            'image_dir': None
        },
        'windows': [0, 2, 10],
        'root_only_reference_window': 2,  # used when windows=[0] only
        'embedding_mode': 'sharded',
        'has_user_degrees': True,
        'has_images': False
    },
    'gaming': {
        'type': 'reddit',
        'paths': {
            'thread_metadata': os.path.join(REDDIT_ROOT, "gaming/metadata/reddit_gaming_metadata.parquet"),
            'thread_posts': os.path.join(REDDIT_ROOT, "gaming/metadata/reddit_gaming_posts.parquet"),
            'user_degrees': None,
            'early_window_dir': os.path.join(REDDIT_ROOT, "gaming/snapshots"),
            'embedding_dir': os.path.join(REDDIT_ROOT, "gaming/embeddings"),
            'output_dir': RESULTS_ROOT,
            'image_dir': os.path.join(REDDIT_ROOT, "gaming/images")
        },
        'windows': [0, 20, 50],
        'root_only_reference_window': 20,
        'embedding_mode': 'single_file',
        'has_user_degrees': False,
        'has_images': True
    },
    'futurology': {
        'type': 'reddit',
        'paths': {
            'thread_metadata': os.path.join(REDDIT_ROOT, "futurology/metadata/reddit_futurology_metadata.parquet"),
            'thread_posts': os.path.join(REDDIT_ROOT, "futurology/metadata/reddit_futurology_posts.parquet"),
            'user_degrees': None,
            'early_window_dir': os.path.join(REDDIT_ROOT, "futurology/snapshots"),
            'embedding_dir': os.path.join(REDDIT_ROOT, "futurology/embeddings"),
            'output_dir': RESULTS_ROOT,
            'image_dir': os.path.join(REDDIT_ROOT, "futurology/images")
        },
        'windows': [0, 90],
        'root_only_reference_window': 90,
        'embedding_mode': 'single_file',
        'has_user_degrees': False,
        'has_images': True
    },
    'ama': {
        'type': 'reddit',
        'paths': {
            'thread_metadata': os.path.join(REDDIT_ROOT, "ama/metadata/reddit_ama_metadata.parquet"),
            'thread_posts': os.path.join(REDDIT_ROOT, "ama/metadata/reddit_ama_posts.parquet"),
            'user_degrees': None,
            'early_window_dir': os.path.join(REDDIT_ROOT, "ama/snapshots"),
            'embedding_dir': os.path.join(REDDIT_ROOT, "ama/embeddings"),
            'output_dir': RESULTS_ROOT,
            'image_dir': os.path.join(REDDIT_ROOT, "ama/images")
        },
        'windows': [0, 30],
        'root_only_reference_window': 15,
        'embedding_mode': 'single_file',
        'has_user_degrees': False,
        'has_images': True
    }
}
