"""
Feature configuration
"""
import config.config as cfg

# Check if user degrees are available for current dataset
HAS_USER_DEGREES = cfg.DATASET_CONF.get('has_user_degrees', True)

# Node-level features
NODE_FEATURES = {
    'content': {
        'text_embedding': {'dim': 384, 'project_to': 32, 'standardize': False},
    },
    'temporal': {
        'log_time_since_root': {'standardize': True},
        'log_time_since_parent': {'standardize': True},
    },
}

# Global features
root_meta_features = {
    'time_sin': {},
    'time_cos': {},
}

if HAS_USER_DEGREES:
    root_meta_features['log_follower_count'] = {'standardize': True}

GLOBAL_FEATURES = {
    'root_metadata': root_meta_features
}

# ============================================================
# TARGET GROUPS — Separate MLP task heads per semantic group
#
# Justification:
#   structural  : Emergent whole-tree properties driven by propagation
#                 dynamics (GNN topology signal dominates).
#   participation: Distinct-user engagement — hybrid of structure and
#                 content appeal. Kept separate from structural because
#                 a wide thread ≠ many unique users (bots, duplicates).
#   engagement  : Root-post-level reception (likes / score) that exists
#                 BEFORE the thread grows. Measures intrinsic content
#                 appeal, not conversation dynamics. Most semantically
#                 distinct group — warrants its own head.
# ============================================================

# Group 1 — Structural diffusion (whole-tree emergent properties)
_STRUCTURAL = ['max_width', 'max_depth', 'structural_virality', 'num_posts']

# Group 2 — Participation (distinct users who engaged)
# num_unique_users is present in Reddit metadata; add for Bluesky if available
#_PARTICIPATION = []
"""if cfg.CURRENT_DATASET_NAME != 'bluesky':
    _PARTICIPATION = ['num_unique_users']"""
_PARTICIPATION = ['num_unique_users']

# Group 3 — Engagement (root post reception only)
# root_score is extracted from the root post's like_count (Bluesky) or score (Reddit)
# in data_loader.py — it is NOT a thread-level aggregate.
_ENGAGEMENT = ['root_score']

# Build ordered dict (order defines tensor column positions)
TARGET_GROUPS = {}
TARGET_GROUPS['structural'] = _STRUCTURAL
if _PARTICIPATION:
    TARGET_GROUPS['participation'] = _PARTICIPATION
TARGET_GROUPS['engagement'] = _ENGAGEMENT

# Flat ordered list — must be kept in sync with TARGET_GROUPS ordering
OUTPUT_TARGETS = [target for targets in TARGET_GROUPS.values() for target in targets]

# Map target name → (group_name, local_index) for interpretable evaluation
TARGET_GROUP_MAP = {
    target: (group, i)
    for group, targets in TARGET_GROUPS.items()
    for i, target in enumerate(targets)
}

TARGET_TRANSFORM = 'log1p'

# Index of root_score in OUTPUT_TARGETS — used to mask its loss at non-final horizons.
ROOT_SCORE_IDX = OUTPUT_TARGETS.index('root_score')

# Mapping: gt column name in future_horizons parquets → OUTPUT_TARGETS target name.
# root_score is absent from intermediate horizons — it is handled separately (masked).
HORIZON_GT_COL_MAP = {
    'gt_max_width':           'max_width',
    'gt_max_depth':           'max_depth',
    'gt_structural_virality': 'structural_virality',
    'gt_num_posts':           'num_posts',
    'gt_num_unique_users':    'num_unique_users',
}