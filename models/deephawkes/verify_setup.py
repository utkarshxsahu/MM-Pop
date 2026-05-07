#!/usr/bin/env python
"""
Verification script to check DeepHawkes setup.
"""

from pathlib import Path

import numpy as np

import config


def check_mark(condition):
    return "✓" if condition else "✗"


def verify_dataset(dataset_name: str) -> bool:
    dataset_config = config.DATASETS[dataset_name]
    paths = dataset_config["paths"]
    windows = dataset_config["windows"]
    split_dir = Path(paths["early_window_dir"]) / "splits"

    print(f"\n{'=' * 80}")
    print(f"Verifying Dataset: {dataset_name.upper()}")
    print(f"{'=' * 80}")

    all_ok = True
    for path_name, path_value in paths.items():
        exists = Path(path_value).exists()
        print(f"{check_mark(exists)} {path_name}: {path_value}")
        all_ok &= exists

    print(f"\nChecking windows: {windows}")
    for window in windows:
        split_window = dataset_config["root_only_split_window"] if window == 0 else window
        if window == 0:
            print(f"\n  Window: 0 minutes (root-only, reuses {split_window}min splits)")
        else:
            print(f"\n  Window: {window} minutes")

        if window > 0:
            edges_path = Path(paths["early_window_dir"]) / f"edges_{window}min.parquet"
            edges_exists = edges_path.exists()
            print(f"    {check_mark(edges_exists)} Edges: {edges_path}")
            all_ok &= edges_exists

        train_path = split_dir / f"train_ids_{split_window}min.npy"
        val_path = split_dir / f"val_ids_{split_window}min.npy"
        test_path = split_dir / f"test_ids_{split_window}min.npy"
        split_ok = train_path.exists() and val_path.exists() and test_path.exists()
        print(f"    {check_mark(split_ok)} Split files: {train_path.name}, {val_path.name}, {test_path.name}")
        all_ok &= split_ok

        if split_ok:
            train_ids = np.load(train_path, allow_pickle=True)
            val_ids = np.load(val_path, allow_pickle=True)
            test_ids = np.load(test_path, allow_pickle=True)
            print(f"      Train: {len(train_ids)}")
            print(f"      Val:   {len(val_ids)}")
            print(f"      Test:  {len(test_ids)}")

    print(f"\nTargets: {config.TARGETS}")
    print(f"Future horizons enabled: {config.USE_FUTURE_HORIZONS}")
    return all_ok


def verify_output_dir() -> bool:
    results_dir = Path(config.RESULTS_BASE_DIR)
    print(f"\n{'=' * 80}")
    print("Verifying Output Configuration")
    print(f"{'=' * 80}")
    print(f"Results base dir: {results_dir.absolute()}")
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        print("✓ Results directory is writable")
        return True
    except Exception as exc:
        print(f"✗ Cannot create results directory: {exc}")
        return False


def main():
    print("\n" + "=" * 80)
    print("DeepHawkes Setup Verification")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Datasets to run: {config.DATASETS_TO_RUN}")
    print(f"  Save checkpoints: {config.SAVE_CHECKPOINTS}")
    print(f"  Debug mode: {config.DEBUG_MODE}")
    print(f"  Future horizons: {config.USE_FUTURE_HORIZONS}")
    print(f"  Seed: {config.SEED}")

    all_ok = verify_output_dir()
    for dataset_name in config.DATASETS_TO_RUN:
        if dataset_name not in config.DATASETS:
            print(f"\n✗ Dataset '{dataset_name}' not found in config.DATASETS")
            all_ok = False
            continue
        all_ok &= verify_dataset(dataset_name)

    print(f"\n{'=' * 80}")
    print("✓ ALL CHECKS PASSED - Ready to train!" if all_ok else "✗ SOME CHECKS FAILED - Please fix the issues above")
    print(f"{'=' * 80}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
