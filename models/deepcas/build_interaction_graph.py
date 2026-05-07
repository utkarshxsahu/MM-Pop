"""
Build dataset-specific interaction graph embeddings.
"""

import argparse
import json
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from joblib import Parallel, delayed

import config


DIMENSIONS = 128
WALK_LENGTH = 20
NUM_WALKS = 5
WINDOW = 10
WORKERS = 12
EPOCHS = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train-split-only user interaction graph embeddings."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(config.DATASETS.keys()),
        default=config.CURRENT_DATASET,
        help="Dataset to build embeddings for. Defaults to config.CURRENT_DATASET.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Build embeddings for every dataset in config.DATASETS.",
    )
    parser.add_argument(
        "--split-window",
        type=int,
        default=None,
        help=(
            "Window whose train_ids_<window>min.npy split should define the training "
            "trees. Defaults to the dataset root-only reference window."
        ),
    )
    parser.add_argument(
        "--all-windows",
        action="store_true",
        help="Build embeddings for every configured split window in each selected dataset.",
    )
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS)
    parser.add_argument("--walk-length", type=int, default=WALK_LENGTH)
    parser.add_argument("--num-walks", type=int, default=NUM_WALKS)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args()


def get_train_ids_for_embedding(split_window):
    train_path = os.path.join(config.SPLIT_DIR, f"train_ids_{split_window}min.npy")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Missing train split for dataset '{config.CURRENT_DATASET}', "
            f"window {split_window}min: {train_path}"
        )
    return np.load(train_path, allow_pickle=True)


def generate_walks_chunk(nodes_chunk, graph_data, walk_length, num_walks, seed):
    """Generate random walks for a chunk of nodes."""
    rng = random.Random(seed)
    walks = []
    for _ in range(num_walks):
        for node in nodes_chunk:
            if node not in graph_data:
                continue

            walk = [node]
            curr = node
            for _ in range(walk_length - 1):
                neighbors = list(graph_data[curr].keys())
                if not neighbors:
                    break
                weights = [graph_data[curr][nbr]["weight"] for nbr in neighbors]
                next_node = rng.choices(neighbors, weights=weights, k=1)[0]
                walk.append(next_node)
                curr = next_node
            walks.append([str(n) for n in walk])
    return walks


def _selected_datasets(args):
    if args.all_datasets:
        return list(config.DATASETS.keys())
    return [args.dataset]


def _selected_split_windows(args):
    if args.split_window is not None:
        return [args.split_window]
    if args.all_windows:
        windows = set(config.WINDOWS)
        if 0 in windows:
            windows.remove(0)
            if config.ROOT_ONLY_REFERENCE_WINDOW is not None:
                windows.add(config.ROOT_ONLY_REFERENCE_WINDOW)
        return sorted(windows)
    return [config.ROOT_ONLY_REFERENCE_WINDOW]


def build_embeddings_for_current_dataset(args, split_window):
    if split_window is None:
        raise ValueError(
            f"Dataset '{config.CURRENT_DATASET}' has no default split window; "
            "pass --split-window explicitly."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    embedding_paths = config.get_embedding_paths_for_window(split_window)
    os.makedirs(embedding_paths["dir"], exist_ok=True)

    print(f"Building interaction graph for dataset: {config.CURRENT_DATASET}")
    print(f"Using train split: train_ids_{split_window}min.npy")
    print(f"Loading posts from: {config.THREAD_POSTS}")
    train_ids = get_train_ids_for_embedding(split_window)
    train_tree_ids = set(train_ids.tolist())
    df = pd.read_parquet(config.THREAD_POSTS)
    df = df[df["tree_id"].isin(train_tree_ids)].copy()
    print(f"Training trees in split: {len(train_tree_ids):,}")
    print(f"Posts after train split filter: {len(df):,}")

    print("Building train-only interaction graph...")
    edge_df = df[["user_id", "replied_author"]].dropna().copy()
    if edge_df.empty:
        raise ValueError(
            f"No user_id/replied_author edges found for {config.CURRENT_DATASET} "
            f"train split {split_window}min."
        )
    edge_df["user_id"] = edge_df["user_id"].astype(str)
    edge_df["replied_author"] = edge_df["replied_author"].astype(str)

    graph = nx.DiGraph()
    interactions = edge_df.groupby(["user_id", "replied_author"]).size().reset_index(name="weight")
    graph.add_weighted_edges_from(interactions.values)

    print(f"Graph Built: {graph.number_of_nodes():,} users, {graph.number_of_edges():,} unique edges.")

    print(f"Generating random walks using {args.workers} cores...")
    nodes = list(graph.nodes())
    chunk_size = max(len(nodes) // args.workers, 1)
    node_chunks = [nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)]
    graph_adj = nx.to_dict_of_dicts(graph)

    results = Parallel(n_jobs=args.workers, backend="loky", verbose=10)(
        delayed(generate_walks_chunk)(
            chunk,
            graph_adj,
            args.walk_length,
            args.num_walks,
            args.seed + chunk_idx,
        )
        for chunk_idx, chunk in enumerate(node_chunks)
    )
    all_walks = [walk for chunk_walks in results for walk in chunk_walks]
    print(f"Generated {len(all_walks):,} walks.")

    print("Training Word2Vec...")
    model = Word2Vec(
        sentences=all_walks,
        vector_size=args.dimensions,
        window=args.window,
        min_count=1,
        sg=1,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.seed,
    )

    print("Saving embeddings and vocabulary...")
    vocab_list = model.wv.index_to_key
    vectors = model.wv.vectors
    unk_vector = np.mean(vectors, axis=0, keepdims=True)
    final_vectors = np.vstack([unk_vector, vectors])

    embeddings_tmp = embedding_paths["embeddings"] + ".tmp"
    vocab_tmp = embedding_paths["vocab"] + ".tmp"
    metadata_path = embedding_paths["metadata"]
    metadata_tmp = metadata_path + ".tmp"

    np.save(embeddings_tmp, final_vectors)

    vocab_map = {"<UNK>": 0}
    for idx, user_id in enumerate(vocab_list):
        vocab_map[user_id] = idx + 1

    with open(vocab_tmp, "w") as f:
        json.dump(vocab_map, f)

    metadata = {
        "dataset": config.CURRENT_DATASET,
        "graph_source": "train_split_user_reply_interaction",
        "train_split_window_minutes": split_window,
        "thread_posts": config.THREAD_POSTS,
        "train_ids_path": os.path.join(config.SPLIT_DIR, f"train_ids_{split_window}min.npy"),
        "num_train_tree_ids": len(train_tree_ids),
        "num_posts_after_filter": int(len(df)),
        "num_interaction_edges_raw": int(len(edge_df)),
        "num_graph_nodes": graph.number_of_nodes(),
        "num_graph_edges": graph.number_of_edges(),
        "dimensions": args.dimensions,
        "walk_length": args.walk_length,
        "num_walks": args.num_walks,
        "window": args.window,
        "workers": args.workers,
        "epochs": args.epochs,
        "seed": args.seed,
        "embedding_path": embedding_paths["embeddings"],
        "vocab_path": embedding_paths["vocab"],
    }
    with open(metadata_tmp, "w") as f:
        json.dump(metadata, f, indent=2)

    os.replace(embeddings_tmp + ".npy", embedding_paths["embeddings"])
    os.replace(vocab_tmp, embedding_paths["vocab"])
    os.replace(metadata_tmp, metadata_path)

    print(f"Saved embeddings to: {embedding_paths['embeddings']}")
    print(f"Saved vocabulary to: {embedding_paths['vocab']}")
    print(f"Saved metadata to: {metadata_path}")


def main():
    args = parse_args()

    for dataset_name in _selected_datasets(args):
        config.set_dataset(dataset_name)
        for split_window in _selected_split_windows(args):
            build_embeddings_for_current_dataset(args, split_window)


if __name__ == "__main__":
    main()
