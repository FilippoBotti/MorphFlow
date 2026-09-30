"""Run: python3 -m torch.distributed.run --standalone --nproc_per_node=2 tools/smoke_prior_balance_ddp.py

CPU/Gloo, tiny model, five alternating split/ordinary updates. Compares global
component gradients, weighted updates and AdamW states with a serial reference.
No real TRELLIS assets or GPU allocation required. Add --bf16 for CPU autocast.
"""
import argparse
import copy
from contextlib import nullcontext
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from modules.prior_gradient_balance import BalanceConfig, PriorGradientBalancer


class AcceleratorStub:
    scaler = None
    def backward(self, loss, **kwargs):
        loss.backward(**kwargs)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Linear(3, 4)
        self.lora = nn.Linear(4, 2)

    def forward(self, x, target, threshold, split):
        def apply(v):
            return self.lora(torch.tanh(self.condition(v)))
        fm = (apply(x) - target).float().square().mean()
        if not split:
            return fm
        z = checkpoint(apply, x + 0.2, use_reentrant=False).float()
        projection = 0.5 * (z - (z.detach() + 0.13)).square().mean()
        guard = torch.relu(threshold - z.square().mean(1).clamp_min(1e-12).sqrt()).square().mean()
        return dict(fm=fm, projection=projection, guard=guard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bf16', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(340)
    net = SmallModel()
    reference = copy.deepcopy(net)
    ddp = torch.nn.parallel.DistributedDataParallel(net, find_unused_parameters=False)
    groups = lambda m: [dict(params=list(m.condition.parameters()), name='condition'),
                        dict(params=list(m.lora.parameters()), name='lora')]
    opt = torch.optim.AdamW(groups(net), lr=1e-3)
    refopt = torch.optim.AdamW(groups(reference), lr=1e-3)
    controller = PriorGradientBalancer(opt, BalanceConfig(warmup_steps=0, ema=0), 70000)
    tol = dict(atol=1e-2, rtol=2e-2) if args.bf16 else dict(atol=3e-6, rtol=3e-5)
    for i in range(6):
        all_x = torch.arange(world * 6, dtype=torch.float32).reshape(world * 2, 3) / 13 + i * 0.1
        all_y = torch.sin(all_x[:, :2])
        thresholds = torch.zeros(world * 2)
        if i == 2:  # Guard active only on one rank: collective branch must agree.
            thresholds[-2:] = 2.0
        x = all_x[rank*2:(rank+1)*2]
        y = all_y[rank*2:(rank+1)*2]
        threshold = thresholds[rank*2:(rank+1)*2]
        active = i % 2 == 0
        opt.zero_grad(set_to_none=True)
        refopt.zero_grad(set_to_none=True)
        context = ddp.no_sync() if active else nullcontext()
        with context:
            with torch.autocast('cpu', enabled=args.bf16, dtype=torch.bfloat16, cache_enabled=False):
                output = ddp(x, y, threshold, active)
            if active:
                _, metrics = controller.backward_parts(output, AcceleratorStub(), ddp, 70000+i)
                weight = metrics['trellis_prior_weight']
                assert abs(metrics['balance/ratio_projection_fm'] - 0.5) < 1e-5
            else:
                output.backward()
                weight = 0.0
        with torch.autocast('cpu', enabled=args.bf16, dtype=torch.bfloat16, cache_enabled=False):
            ref_out = reference(all_x, all_y, thresholds, active)
        if active:
            (ref_out['fm'] + weight*ref_out['projection'] + 0.1*ref_out['guard']).backward()
        else:
            ref_out.backward()
        for p, q in zip(net.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, **tol)
        opt.step()
        refopt.step()
        for p, q in zip(net.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, **tol)
        for actual, expected in zip(opt.state.values(), refopt.state.values()):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(actual[key], expected[key], **tol)
        if i == 3:
            state = controller.state_dict()
            controller = PriorGradientBalancer(opt, BalanceConfig(warmup_steps=0, ema=0), 70000)
            controller.load_state_dict(state)
        if rank == 0:
            print('PASS update=%d split=%s guard=%s' % (i+1, active, i == 2), flush=True)
    dist.barrier()
    if rank == 0:
        print('OK: global gradients, alternating DDP updates, guard branch and controller resume', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
