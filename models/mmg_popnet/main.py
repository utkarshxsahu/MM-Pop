"""
Main entry point for GNN experiments.

Single GPU:
    python main.py

4-GPU DDP:
    torchrun --nproc_per_node=4 main.py
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

        # Override cfg.DEVICE so every downstream call uses this rank's GPU
        import config.config as cfg
        cfg.DEVICE             = torch.device(f'cuda:{local_rank}')
        cfg.DEVICE_TRANSFORMER = cfg.DEVICE
        cfg.DEVICE_GNN         = cfg.DEVICE

        if rank == 0:
            print(f"[DDP] {world_size} processes initialised. "
                  f"This rank={rank} on cuda:{local_rank}")
    else:
        import config.config as cfg

    from training.experiment_runner import run_all_experiments
    run_all_experiments(rank=rank, world_size=world_size)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
