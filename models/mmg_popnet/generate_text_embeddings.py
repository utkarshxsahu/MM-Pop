#!/usr/bin/env python3
"""
Generate MiniLM text embeddings for MMG-PopNet.

Outputs are written into the dataset-local embeddings folders:
  datasets/bluesky/embeddings/
  datasets/reddit/{gaming,futurology,ama}/embeddings/

Reddit uses one embeddings384_fp16.npy plus index.parquet per subreddit.
Bluesky uses sharded embeddings384_fp16_shard_*.npy plus index_shard_*.parquet.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_SHARD_ROWS = 1_000_000

REDDIT_DATASETS = {
    "gaming": DATASETS_ROOT / "reddit" / "gaming" / "metadata" / "reddit_gaming_posts.parquet",
    "futurology": DATASETS_ROOT / "reddit" / "futurology" / "metadata" / "reddit_futurology_posts.parquet",
    "ama": DATASETS_ROOT / "reddit" / "ama" / "metadata" / "reddit_ama_posts.parquet",
}
BLUESKY_POSTS = DATASETS_ROOT / "bluesky" / "metadata" / "thread_posts_with_all_labels2.parquet"


def setup_model(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return SentenceTransformer(model_name, device=device)


def _ensure_columns(parquet_path: Path, required_columns: list[str]) -> None:
    schema = pq.ParquetFile(parquet_path).schema_arrow
    missing = [col for col in required_columns if col not in schema.names]
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {missing}")


def process_reddit_subreddit(
    model: SentenceTransformer,
    subreddit_name: str,
    input_file: Path,
    output_dir: Path,
    batch_size: int,
    model_name: str,
) -> dict:
    print(f"\n{'=' * 70}")
    print(f"Processing Reddit: {subreddit_name}")
    print(f"{'=' * 70}")

    if not input_file.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_file}")
    _ensure_columns(input_file, ["post_id", "text"])
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / "embeddings384_fp16.npy"
    idx_path = output_dir / "index.parquet"
    metadata_path = output_dir / "metadata.json"

    pf = pq.ParquetFile(input_file)
    all_embeddings = []
    all_post_ids = []
    total_rows = 0

    for batch in tqdm(pf.iter_batches(columns=["post_id", "text"]), desc=subreddit_name):
        df = batch.to_pandas()
        df["post_id"] = df["post_id"].astype(str)
        df["text"] = df["text"].fillna("").astype(str)

        embeddings = model.encode(
            df["text"].tolist(),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float16, copy=False)

        all_embeddings.append(embeddings)
        all_post_ids.extend(df["post_id"].tolist())
        total_rows += len(df)

    if not all_embeddings:
        raise ValueError(f"No rows found in {input_file}")

    final_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(emb_path, final_embeddings, allow_pickle=False)

    index_df = pd.DataFrame(
        {
            "post_id": all_post_ids,
            "row_idx": np.arange(len(all_post_ids), dtype=np.int32),
        }
    )
    n_duplicates = int(index_df["post_id"].duplicated().sum())
    if n_duplicates:
        print(f"WARNING: Found {n_duplicates:,} duplicate post_ids; keeping first index occurrence.")
        index_df = index_df.drop_duplicates(subset=["post_id"], keep="first")

    table = pa.Table.from_pandas(index_df, preserve_index=False)
    pq.write_table(table, idx_path, compression="zstd")

    metadata = {
        "dataset": "reddit",
        "subreddit": subreddit_name,
        "input_file": str(input_file),
        "total_rows": int(total_rows),
        "total_posts": int(len(index_df)),
        "embedding_dim": int(final_embeddings.shape[1]),
        "embedding_dtype": str(final_embeddings.dtype),
        "model": model_name,
        "duplicates_found": n_duplicates,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Completed {subreddit_name}: {final_embeddings.shape} -> {output_dir}")
    return metadata


def _buffered_rows(buffer_embs: list[np.ndarray]) -> int:
    return sum(arr.shape[0] for arr in buffer_embs)


def _flush_bluesky_shard(
    out_dir: Path,
    shard_id: int,
    shard_keys: list[dict],
    shard_embs: np.ndarray,
) -> dict:
    emb_name = f"embeddings384_fp16_shard_{shard_id:05d}.npy"
    idx_name = f"index_shard_{shard_id:05d}.parquet"
    emb_path = out_dir / emb_name
    idx_path = out_dir / idx_name

    np.save(emb_path, shard_embs.astype(np.float16, copy=False), allow_pickle=False)
    index_df = pd.DataFrame(shard_keys)
    index_df["shard"] = np.int32(shard_id)
    index_df["row"] = np.arange(len(index_df), dtype=np.int32)
    table = pa.Table.from_pandas(index_df, preserve_index=False)
    pq.write_table(table, idx_path, compression="zstd")

    return {
        "shard": shard_id,
        "rows": int(len(index_df)),
        "embedding_file": emb_name,
        "index_file": idx_name,
    }


def process_bluesky(
    model: SentenceTransformer,
    input_file: Path,
    output_dir: Path,
    batch_size: int,
    shard_rows: int,
    model_name: str,
) -> dict:
    print(f"\n{'=' * 70}")
    print("Processing Bluesky")
    print(f"{'=' * 70}")

    if not input_file.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_file}")
    _ensure_columns(input_file, ["tree_id", "post_id", "user_id", "text"])
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "metadata.json"
    pf = pq.ParquetFile(input_file)

    shard_id = 0
    buffer_keys: list[dict] = []
    buffer_embs: list[np.ndarray] = []
    total_rows = 0
    shard_records = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for batch in tqdm(
            pf.iter_batches(columns=["tree_id", "post_id", "user_id", "text"]),
            desc="bluesky",
        ):
            df = batch.to_pandas()
            df["tree_id"] = df["tree_id"].astype(np.int64)
            df["post_id"] = df["post_id"].astype(str)
            df["user_id"] = df["user_id"].astype(str)
            df["text"] = df["text"].fillna("").astype(str)

            emb384 = model.encode(
                df["text"].tolist(),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype(np.float16, copy=False)

            buffer_keys.extend(
                {"tree_id": int(t), "post_id": str(p), "user_id": str(u)}
                for t, p, u in zip(df["tree_id"], df["post_id"], df["user_id"])
            )
            buffer_embs.append(emb384)
            total_rows += len(df)

            while _buffered_rows(buffer_embs) >= shard_rows:
                all_embs = np.concatenate(buffer_embs, axis=0)
                shard_embs = all_embs[:shard_rows]
                leftover_embs = all_embs[shard_rows:]
                shard_keys = buffer_keys[:shard_rows]
                buffer_keys = buffer_keys[shard_rows:]
                buffer_embs = [leftover_embs] if leftover_embs.size else []

                rec = _flush_bluesky_shard(output_dir, shard_id, shard_keys, shard_embs)
                manifest.write(json.dumps(rec) + "\n")
                manifest.flush()
                shard_records.append(rec)
                shard_id += 1

        if buffer_keys:
            final_embs = np.concatenate(buffer_embs, axis=0)
            rec = _flush_bluesky_shard(output_dir, shard_id, buffer_keys, final_embs)
            manifest.write(json.dumps(rec) + "\n")
            manifest.flush()
            shard_records.append(rec)

    metadata = {
        "dataset": "bluesky",
        "input_file": str(input_file),
        "total_posts": int(total_rows),
        "embedding_dim": 384,
        "embedding_dtype": "float16",
        "model": model_name,
        "shard_rows": int(shard_rows),
        "num_shards": int(len(shard_records)),
        "manifest": str(manifest_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Completed Bluesky: {total_rows:,} rows, {len(shard_records)} shard(s) -> {output_dir}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text embeddings for MMG-PopNet.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "bluesky", "gaming", "futurology", "ama"],
        help="Datasets to process.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--model-name", default=MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = ["bluesky", "gaming", "futurology", "ama"] if "all" in args.datasets else args.datasets

    print("\n" + "=" * 70)
    print("MMG-PopNet Text Embedding Generation")
    print("=" * 70)
    print(f"Datasets root: {DATASETS_ROOT}")
    print(f"Datasets: {selected}")

    model = setup_model(args.model_name)
    results = {}

    if "bluesky" in selected:
        results["bluesky"] = process_bluesky(
            model=model,
            input_file=BLUESKY_POSTS,
            output_dir=DATASETS_ROOT / "bluesky" / "embeddings",
            batch_size=args.batch_size,
            shard_rows=args.shard_rows,
            model_name=args.model_name,
        )

    for subreddit in ["gaming", "futurology", "ama"]:
        if subreddit not in selected:
            continue
        results[subreddit] = process_reddit_subreddit(
            model=model,
            subreddit_name=subreddit,
            input_file=REDDIT_DATASETS[subreddit],
            output_dir=DATASETS_ROOT / "reddit" / subreddit / "embeddings",
            batch_size=args.batch_size,
            model_name=args.model_name,
        )

    summary_path = DATASETS_ROOT / "embedding_generation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAll done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
