# MMG-Pop

Code repository for the MMG-Pop (Multimodal Graph-based Popularity Prediction) benchmark.

## Abstract

Social media popularity prediction aims to forecast the future reach or influence of online content from early-stage observations.
Accurate prediction enables key downstream applications, such as advertising optimization and strategic content planning by users, creators, and platforms.
Despite substantial progress, existing popularity prediction works often fail to jointly consider multimodal content and temporal social interaction signals. Moreover, the literature remains highly fragmented across datasets, modalities, observation windows, prediction targets, and evaluation protocols. This fragmentation prevents fair comparison and obscures a systematic understanding of how textual, visual, temporal, and interaction-based signals jointly shape popularity dynamics.
To address these challenges, we introduce **MMG-Pop**, the first **Multi-modal Graph-based Popularity Prediction** benchmark, which unifies datasets, modalities, temporal interaction signals, and representative baselines under a standardized evaluation protocol. Furthermore, we propose **MMG-PopNet**, a unified multi-modal graph-based network that jointly models multimodal signals and graph-structured social interactions.
Extensive experiments on **MMG-Pop**, comprising four datasets across **Bluesky** and **Reddit**, demonstrate the superior performance of **MMG-PopNet** and yield new insights into cross-platform training generalization, multi-task prediction benefits, multi-modality contributions, and LLM prediction limitations.
These findings establish a unified foundation for future research on social dynamics modeling and intervention under heterogeneous modalities and socially aware agentic ecosystem paradigms.


Access dataset: https://huggingface.co/datasets/anonymoususer54829/MMG-Pop

## Data Layout

Place downloaded datasets under `datasets/`:

```text
datasets/
  bluesky/
    metadata/
    snapshots/
      splits/
      future_horizons/
    embeddings/
  reddit/
    gaming/
      metadata/
      snapshots/
        splits/
        future_horizons/
      embeddings/
      images/
    futurology/
      metadata/
      snapshots/
        splits/
        future_horizons/
      embeddings/
      images/
    ama/
      metadata/
      snapshots/
        splits/
        future_horizons/
      embeddings/
      images/
```

The model configs are repo-relative and expect these filenames:

- Bluesky metadata: `thread_metadata_updated_added.parquet`, `thread_posts_with_all_labels2.parquet`, `user_degrees.csv`
- Reddit metadata: `reddit_<subreddit>_metadata.parquet`, `reddit_<subreddit>_posts.parquet`
- Snapshots: `edges_<window>min.parquet`, `meta_<window>min.parquet`
- Splits: `train_ids_<window>min.npy`, `val_ids_<window>min.npy`, `test_ids_<window>min.npy`
- Future horizons: `gt_4h.parquet`, `gt_8h.parquet`, `gt_16h.parquet`, `gt_24h.parquet`, `valid_mask.parquet`

Results are written to `results/<model>/run...`.

## MMG-PopNet Setup

The recommended setup is the conda environment in `models/mmg_popnet/environment.yml`:

```bash
cd models/mmg_popnet
conda env create -f environment.yml
conda activate soc_analysis2
```

A pip-oriented dependency list is also provided at `models/mmg_popnet/requirements.txt`. PyTorch Geometric wheels are CUDA/PyTorch-version specific, so prefer the conda file when using GPU.

## Generate Text Embeddings

In MMG-PopNet, MLP uses frozen MiniLM post embeddings from each dataset's `embeddings/` folder. `MMG-PopNet` trains from tokenized text directly and does not require these precomputed embeddings.

Generate all embeddings:

```bash
cd models/mmg_popnet
python generate_text_embeddings.py --datasets all
```

Generate a subset:

```bash
python generate_text_embeddings.py --datasets gaming futurology ama
python generate_text_embeddings.py --datasets bluesky --shard-rows 1000000
```

Outputs:

- Reddit: `datasets/reddit/<subreddit>/embeddings/embeddings384_fp16.npy`, `index.parquet`, `metadata.json`
- Bluesky: `datasets/bluesky/embeddings/embeddings384_fp16_shard_00000.npy`, `index_shard_00000.parquet`, `manifest.jsonl`, `metadata.json`

## Future Horizons

Future-horizon mode expands supervision from final-only prediction to `4h`, `8h`, `16h`, `24h`, and `final`. Intermediate horizons supervise structural targets; `root_score` remains final-only.

If horizon files are not already downloaded, generate them after snapshots/splits are present:

```bash
cd models/mmg_popnet
python generate_future_horizons.py --dataset all
```

The generated files go to each dataset's `snapshots/future_horizons/` folder. Runners with future-horizon support load these files when their config enables future horizons.

## Run MMG-PopNet

MMG-PopNet uses `models/mmg_popnet/config/config.py` for default datasets, windows, model selection, image settings, and future-horizon settings.

Use MMG-PopNet (**Note:** In the codebase, **MMG-PopNet** is referred to as `TextGraphSAGE`. Therefore, any occurrence of `text_graphsage` corresponds to **MMG-PopNet**.)

```python
MODELS_TO_RUN = ["text_graphsage"]
```

Use MLP:

```python
MODELS_TO_RUN = ["mlp"]
```

Run:

```bash
cd models/mmg_popnet
python main.py
```

For DDP:

```bash
torchrun --nproc_per_node=4 main.py
```

Outputs go to `results/mmg_popnet/run_<timestamp>/<dataset>/...`, with `experiment.log` files and per-run JSON summaries.

## Foundational MMG-PopNet

The foundational runner trains one model across all configured datasets and observation-window slots, then reports per-dataset and per-slot metrics.

```bash
cd models/mmg_popnet
python main_foundational.py
```

DDP:

```bash
torchrun --nproc_per_node=4 main_foundational.py
```

Outputs go to `results/mmg_popnet/foundational/run_<timestamp>/`.

## Modality Ablation

Modality ablation runs MMG-PopNet variants such as `no_image`, `no_topology`, and `no_temporal`.

```bash
cd models/mmg_popnet
python modality_ablation_runner.py
```

Useful overrides:

```bash
python modality_ablation_runner.py --variants no_image,no_topology
python modality_ablation_runner.py --datasets gaming
python modality_ablation_runner.py --windows gaming:20,50
```

Outputs go to `results/mmg_popnet/ablation_modality/`.

## Run Baseline Models

CasSeqGCN:

```bash
cd models/casseqgcn
python experiment_runner.py
```

DeepCas:

```bash
cd models/deepcas
python main.py
```

DeepHawkes:

```bash
cd models/deephawkes
python train.py
```

Graph-LSTM:

```bash
cd models/graphlstm
python graph_lstm_train.py
```

All baseline configs now point at `datasets/` and write to `results/<model>/`.
