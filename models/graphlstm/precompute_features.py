"""
Precompute and cache per-node structural features and token IDs.
"""

import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import config
from data_loader import load_early_window_data, load_full_data


def get_cache_root(dataset_name=None):
    return os.path.join(config.get_output_root(dataset_name, config.MODE_STANDARD), "graph_lstm_cache")


def get_cache_dir(window_tag=None, dataset_name=None):
    base = get_cache_root(dataset_name)
    if window_tag is None:
        return base
    return os.path.join(base, config.get_window_label(window_tag))


def _build_children(edges_df, time_map, node_set):
    children = defaultdict(list)
    parent_of = {}
    for _, row in edges_df.iterrows():
        pid, rto = row["post_id"], row["reply_to"]
        if pd.notna(rto) and pid in node_set and rto in node_set:
            children[rto].append(pid)
            parent_of[pid] = rto
    for parent in children:
        children[parent].sort(key=lambda x: time_map.get(x, 0))
    return dict(children), parent_of


def _subtree_stats(nodes, children):
    child_set = {child for child_list in children.values() for child in child_list}
    roots = [node for node in nodes if node not in child_set]
    size = {node: 1 for node in nodes}
    height = {node: 0 for node in nodes}

    for root in roots:
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                for child in children.get(node, []):
                    size[node] += size[child]
                    height[node] = max(height[node], height[child] + 1)
            else:
                stack.append((node, True))
                for child in children.get(node, []):
                    stack.append((child, False))
    return size, height


def compute_tree_features(tree_id, edges_df, posts_df, root_author):
    if posts_df.empty:
        return pd.DataFrame()

    early_nodes = set(edges_df["post_id"].tolist()) if not edges_df.empty else set(posts_df["post_id"].tolist())
    posts_df = posts_df[posts_df["post_id"].isin(early_nodes)].copy()
    if posts_df.empty:
        return pd.DataFrame()

    posts_idx = posts_df.set_index("post_id")
    nodes = list(posts_idx.index)
    node_set = set(nodes)

    time_map = posts_idx["time_since_root"].to_dict()
    children, parent_of = _build_children(edges_df, time_map, node_set) if not edges_df.empty else ({}, {})
    subtree_size, subtree_height = _subtree_stats(nodes, children) if nodes else ({}, {})

    num_siblings = {}
    for node in nodes:
        parent = parent_of.get(node)
        siblings = children.get(parent, []) if parent else []
        num_siblings[node] = max(len(siblings) - 1, 0)

    num_children = {node: len(children.get(node, [])) for node in nodes}
    times = np.array([time_map.get(node, 0) for node in nodes], dtype=float)
    times_sorted = np.sort(times)
    num_prev = np.searchsorted(times_sorted, times, side="left")
    num_later = len(nodes) - np.searchsorted(times_sorted, times, side="right")

    user_counts = posts_idx["user_id"].value_counts().to_dict() if "user_id" in posts_idx.columns else {}
    nc_arr = np.array([num_children[node] for node in nodes], dtype=float)
    ns_arr = np.array([subtree_size.get(node, 1) for node in nodes], dtype=float)
    nc_mean = nc_arr.mean() if len(nc_arr) else 0.0
    ns_mean = ns_arr.mean() if len(ns_arr) else 0.0
    nc_ranks = np.argsort(np.argsort(nc_arr)) + 1 if len(nc_arr) else np.array([])
    ns_ranks = np.argsort(np.argsort(ns_arr)) + 1 if len(ns_arr) else np.array([])

    rows = []
    for idx, node in enumerate(nodes):
        post = posts_idx.loc[node]
        user_id = str(post["user_id"]) if "user_id" in posts_idx.columns else ""
        rows.append(
            {
                "tree_id": tree_id,
                "post_id": node,
                "time_since_root": float(post.get("time_since_root", 0) or 0),
                "log_time_since_root": float(post.get("log_time_since_root", 0) or 0),
                "time_since_parent": float(post.get("time_since_parent", 0) or 0),
                "log_time_since_parent": float(post.get("log_time_since_parent", 0) or 0),
                "num_previous": float(num_prev[idx]),
                "num_later": float(num_later[idx]),
                "is_op": float(user_id == str(root_author)),
                "author_count": float(user_counts.get(user_id, 0)),
                "depth": float(post.get("depth", 0) or 0),
                "num_siblings": float(num_siblings.get(node, 0)),
                "num_children": float(nc_arr[idx]) if len(nc_arr) else 0.0,
                "subtree_height": float(subtree_height.get(node, 0)),
                "subtree_size": float(ns_arr[idx]) if len(ns_arr) else 1.0,
                "norm_children_mean": float((nc_arr[idx] if len(nc_arr) else 0.0) - nc_mean),
                "norm_children_rank": float((nc_arr[idx] if len(nc_arr) else 0.0) / np.sqrt(nc_ranks[idx])) if len(nc_ranks) else 0.0,
                "norm_subtree_mean": float((ns_arr[idx] if len(ns_arr) else 1.0) - ns_mean),
                "norm_subtree_rank": float((ns_arr[idx] if len(ns_arr) else 1.0) / np.sqrt(ns_ranks[idx])) if len(ns_ranks) else 1.0,
            }
        )
    return pd.DataFrame(rows)


def compute_root_only_features(posts_df, root_authors):
    root_posts = posts_df[posts_df["post_type"] == "root"].copy()
    rows = []
    for _, post in root_posts.iterrows():
        tree_id = post["tree_id"]
        user_id = str(post.get("user_id", ""))
        rows.append(
            {
                "tree_id": tree_id,
                "post_id": post["post_id"],
                "time_since_root": 0.0,
                "log_time_since_root": 0.0,
                "time_since_parent": 0.0,
                "log_time_since_parent": 0.0,
                "num_previous": 0.0,
                "num_later": 0.0,
                "is_op": float(user_id == str(root_authors.get(tree_id, ""))),
                "author_count": 1.0,
                "depth": 0.0,
                "num_siblings": 0.0,
                "num_children": 0.0,
                "subtree_height": 0.0,
                "subtree_size": 1.0,
                "norm_children_mean": 0.0,
                "norm_children_rank": 0.0,
                "norm_subtree_mean": 0.0,
                "norm_subtree_rank": 1.0,
            }
        )
    return pd.DataFrame(rows)


def build_vocabulary(posts_df, min_freq=10):
    counter = Counter()
    for text in posts_df["text"].dropna():
        counter.update(str(text).lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def tokenize_posts(posts_df, vocab, max_len=100):
    unk_id = vocab["<UNK>"]
    return {
        row["post_id"]: [vocab.get(token, unk_id) for token in str(row.get("text", "") or "").lower().split()[:max_len]]
        for _, row in posts_df.iterrows()
    }


def precompute_window(window_tag, full_posts, full_metadata, dataset_name=None):
    cache_dir = get_cache_dir(window_tag, dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    feat_path = os.path.join(cache_dir, "node_features.parquet")

    if os.path.exists(feat_path):
        print(f"  [{config.get_window_label(window_tag)}] Node features already cached - skipping.")
        return

    if window_tag == config.WINDOW_ROOT_ONLY:
        valid_trees = set(full_metadata["tree_id"].unique())
        root_authors = full_metadata.set_index("tree_id")["root_author"].to_dict() if "root_author" in full_metadata.columns else {}
        feat_df = compute_root_only_features(
            full_posts[full_posts["tree_id"].isin(valid_trees)].copy(),
            root_authors,
        )
        if feat_df.empty:
            print(f"  [{config.get_window_label(window_tag)}] WARNING: No root-only features computed.")
            return
        feat_df.to_parquet(feat_path, index=False)
        print(f"  [{config.get_window_label(window_tag)}] Saved {len(feat_df)} node rows -> {feat_path}")
        return

    edges_df, meta_df = load_early_window_data(window_tag, dataset_name)
    valid_trees = set(meta_df["tree_id"].unique())
    posts_by_tree = dict(tuple(full_posts[full_posts["tree_id"].isin(valid_trees)].groupby("tree_id")))
    root_authors = (
        full_metadata[full_metadata["tree_id"].isin(valid_trees)]
        .set_index("tree_id")["root_author"]
        .to_dict()
        if "root_author" in full_metadata.columns
        else {}
    )

    print(f"  [{config.get_window_label(window_tag)}] Computing features for {len(valid_trees)} trees...")
    all_feats = []
    for tree_id, tree_edges in edges_df.groupby("tree_id"):
        feat_df = compute_tree_features(tree_id, tree_edges, posts_by_tree.get(tree_id, pd.DataFrame()), root_authors.get(tree_id, ""))
        if not feat_df.empty:
            all_feats.append(feat_df)

    if all_feats:
        output = pd.concat(all_feats, ignore_index=True)
        output.to_parquet(feat_path, index=False)
        print(f"  [{config.get_window_label(window_tag)}] Saved {len(output)} node rows -> {feat_path}")
    else:
        print(f"  [{config.get_window_label(window_tag)}] WARNING: No features computed.")


def precompute_vocab_and_tokens(full_posts, windows, dataset_name=None):
    vocab_path = os.path.join(get_cache_dir(None, dataset_name), "vocab.json")
    os.makedirs(get_cache_dir(None, dataset_name), exist_ok=True)

    if os.path.exists(vocab_path):
        print("  Vocabulary already cached - loading.")
        with open(vocab_path) as handle:
            vocab = json.load(handle)
    else:
        print("  Building vocabulary from all posts...")
        vocab = build_vocabulary(full_posts)
        with open(vocab_path, "w") as handle:
            json.dump(vocab, handle)
        print(f"  Vocab size: {len(vocab):,}")

    for window_tag in windows:
        cache_dir = get_cache_dir(window_tag, dataset_name)
        os.makedirs(cache_dir, exist_ok=True)
        tok_path = os.path.join(cache_dir, "token_ids.pkl")

        if os.path.exists(tok_path):
            print(f"  [{config.get_window_label(window_tag)}] Token IDs already cached - skipping.")
            continue

        if window_tag == config.WINDOW_ROOT_ONLY:
            posts_filtered = full_posts[full_posts["post_type"] == "root"].copy()
        else:
            edges_df, _ = load_early_window_data(window_tag, dataset_name)
            window_nodes = set(edges_df["post_id"].unique())
            posts_filtered = full_posts[full_posts["post_id"].isin(window_nodes)].copy()

        token_ids = tokenize_posts(posts_filtered, vocab)
        with open(tok_path, "wb") as handle:
            pickle.dump(token_ids, handle)
        print(f"  [{config.get_window_label(window_tag)}] Saved {len(token_ids):,} token sequences -> {tok_path}")

    return vocab


def main():
    for dataset_name in config.get_requested_datasets():
        config.set_active_dataset(dataset_name)
        print(f"Dataset: {dataset_name.upper()}")
        print("Loading full data (once)...")
        full_metadata, full_posts, _ = load_full_data(dataset_name)

        windows = config.get_requested_windows(dataset_name)
        print("\n--- Structural features ---")
        for window_tag in windows:
            precompute_window(window_tag, full_posts, full_metadata, dataset_name)

        print("\n--- Vocabulary + token IDs ---")
        precompute_vocab_and_tokens(full_posts, windows, dataset_name)

    print("\nAll features cached.")


if __name__ == "__main__":
    main()
