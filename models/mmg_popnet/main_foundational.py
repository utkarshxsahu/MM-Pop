"""
Entry point for foundational model experiments.

Single GPU:
    python main_foundational.py

4-GPU DDP:
    torchrun --nproc_per_node=4 main_foundational.py
"""

import os
import torch
import torch.distributed as dist


def main():
    rank       = int(os.environ.get('RANK',       0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        import config.config as cfg
        cfg.DEVICE             = torch.device(f'cuda:{local_rank}')
        cfg.DEVICE_TRANSFORMER = cfg.DEVICE
        cfg.DEVICE_GNN         = cfg.DEVICE

        if rank == 0:
            print(f"[DDP] {world_size} processes initialised. "
                  f"This rank={rank} on cuda:{local_rank}")
    else:
        import config.config as cfg

    if cfg.USE_TRAINING_SEED:
        from utils.reproducibility import seed_everything
        seed_everything(cfg.TRAINING_SEED)
        if rank == 0:
            print(f"[seed] foundational training seed = {cfg.TRAINING_SEED}")

    # foundational_runner.py lives at root level (same as this file)
    from foundational_runner import run_foundational_experiments
    run_foundational_experiments(rank=rank, world_size=world_size)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
