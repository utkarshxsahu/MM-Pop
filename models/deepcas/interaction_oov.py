import json
import pandas as pd
import numpy as np
from collections import Counter
import config

def analyze_vocabulary_coverage(dataset_name, split_window=None):
    """Check what % of users in conversation trees are in the interaction graph."""
    
    print(f"\n{'='*80}")
    print(f"VOCABULARY COVERAGE ANALYSIS: {dataset_name}")
    print(f"{'='*80}")
    
    # Load interaction graph vocabulary
    try:
        config.set_dataset(dataset_name)
        split_window = split_window or config.ROOT_ONLY_REFERENCE_WINDOW
        vocab_path = config.get_embedding_paths_for_window(split_window)["vocab"]
        with open(vocab_path, 'r') as f:
            user_to_idx = json.load(f)
        interaction_graph_users = set(user_to_idx.keys()) - {'<UNK>'}
        print(f"Interaction graph vocabulary size: {len(interaction_graph_users):,}")
    except FileNotFoundError:
        print(f"ERROR: {vocab_path} not found. Run build_interaction_graph.py first!")
        return
    
    # Load conversation tree posts
    posts_path = config.THREAD_POSTS
    print(f"Loading posts from: {posts_path}")
    posts_df = pd.read_parquet(posts_path)
    
    # Get all users in conversation trees
    tree_users = set(posts_df['user_id'].astype(str).unique())
    print(f"Unique users in conversation trees: {len(tree_users):,}")
    
    # Check coverage
    users_in_vocab = tree_users & interaction_graph_users
    users_oov = tree_users - interaction_graph_users
    
    coverage_pct = len(users_in_vocab) / len(tree_users) * 100
    
    print(f"\n--- Coverage Statistics ---")
    print(f"Users in vocabulary: {len(users_in_vocab):,} ({coverage_pct:.2f}%)")
    print(f"Users OOV (will map to <UNK>): {len(users_oov):,} ({100-coverage_pct:.2f}%)")
    
    # Analyze OOV impact on data
    posts_df['user_id_str'] = posts_df['user_id'].astype(str)
    posts_df['is_oov'] = posts_df['user_id_str'].isin(users_oov)
    
    total_posts = len(posts_df)
    oov_posts = posts_df['is_oov'].sum()
    oov_posts_pct = oov_posts / total_posts * 100
    
    print(f"\n--- Impact on Posts ---")
    print(f"Total posts: {total_posts:,}")
    print(f"Posts by OOV users: {oov_posts:,} ({oov_posts_pct:.2f}%)")
    print(f"Posts by known users: {total_posts - oov_posts:,} ({100-oov_posts_pct:.2f}%)")
    
    # Check if OOV users are just rare users
    user_post_counts = posts_df['user_id_str'].value_counts()
    oov_user_post_counts = user_post_counts[user_post_counts.index.isin(users_oov)]
    
    print(f"\n--- OOV User Activity ---")
    print(f"OOV users with 1 post: {(oov_user_post_counts == 1).sum():,}")
    print(f"OOV users with 2-5 posts: {((oov_user_post_counts >= 2) & (oov_user_post_counts <= 5)).sum():,}")
    print(f"OOV users with 6+ posts: {(oov_user_post_counts > 5).sum():,}")
    print(f"Mean posts per OOV user: {oov_user_post_counts.mean():.2f}")
    print(f"Mean posts per known user: {user_post_counts[user_post_counts.index.isin(users_in_vocab)].mean():.2f}")
    
    # Check by tree
    if 'tree_id' in posts_df.columns:
        tree_oov_stats = posts_df.groupby('tree_id')['is_oov'].agg(['sum', 'count'])
        tree_oov_stats['oov_pct'] = tree_oov_stats['sum'] / tree_oov_stats['count'] * 100
        
        print(f"\n--- Per-Tree OOV Analysis ---")
        print(f"Trees with 0% OOV posts: {(tree_oov_stats['oov_pct'] == 0).sum():,}")
        print(f"Trees with 1-25% OOV posts: {((tree_oov_stats['oov_pct'] > 0) & (tree_oov_stats['oov_pct'] <= 25)).sum():,}")
        print(f"Trees with 26-50% OOV posts: {((tree_oov_stats['oov_pct'] > 25) & (tree_oov_stats['oov_pct'] <= 50)).sum():,}")
        print(f"Trees with 50%+ OOV posts: {(tree_oov_stats['oov_pct'] > 50).sum():,}")
        print(f"Mean OOV % per tree: {tree_oov_stats['oov_pct'].mean():.2f}%")
    
    return {
        'dataset': dataset_name,
        'split_window': split_window,
        'total_users_in_trees': len(tree_users),
        'users_in_vocab': len(users_in_vocab),
        'users_oov': len(users_oov),
        'coverage_pct': coverage_pct,
        'oov_posts_pct': oov_posts_pct
    }

# Run for all datasets
if __name__ == '__main__':
    datasets = ['gaming', 'futurology', 'ama', 'bluesky']
    
    results = []
    for dataset in datasets:
        try:
            config.set_dataset(dataset)
            windows = sorted(set(config.WINDOWS) - {0})
            for window in windows:
                result = analyze_vocabulary_coverage(dataset, split_window=window)
                results.append(result)
        except Exception as e:
            print(f"\nERROR processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL DATASETS")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} {'Window':<8} {'Total Users':<12} {'In Vocab':<12} {'OOV':<12} {'Coverage %':<12} {'OOV Posts %':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<15} {str(r['split_window'])+'min':<8} {r['total_users_in_trees']:<12,} {r['users_in_vocab']:<12,} "
              f"{r['users_oov']:<12,} {r['coverage_pct']:<12.2f} {r['oov_posts_pct']:<12.2f}")
