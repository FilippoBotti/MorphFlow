"""Test GPU/import/NCCL attraverso lo stesso launcher usato per il training."""

import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        expected_world = int(os.environ["MF_NUM_PROCESSES"])
        assert world_size == expected_world, (world_size, expected_world)

        # Un vettore abbastanza grande da esercitare anche la comunicazione GPU.
        values = torch.full((1024 * 1024,), float(rank + 1), device="cuda")
        dist.all_reduce(values)
        expected = world_size * (world_size + 1) / 2
        assert torch.all(values == expected).item(), "Risultato all_reduce errato"

        # Verifica che gli import richiesti dal training funzionino su ogni GPU.
        from train import build_parser

        build_parser()
        hosts = [None] * world_size
        dist.all_gather_object(hosts, socket.gethostname())
        assert len(set(hosts)) == int(os.environ["MF_NODE_COUNT"]), hosts
        print(
            f"OK host={socket.gethostname()} rank={rank}/{world_size} "
            f"local_rank={local_rank} gpu={torch.cuda.get_device_name(local_rank)} "
            f"torch={torch.__version__} cuda={torch.version.cuda}",
            flush=True,
        )
        dist.barrier()
        if rank == 0:
            print(f"MULTINODE CHECK PASSED: {len(set(hosts))} nodi, {world_size} GPU.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
