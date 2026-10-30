"""Real SS/SLat projection coverage and strict legacy checkpoint loading."""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import unittest

os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'xformers'
import torch
from torch import nn
from models.lora import add_lora_to_attention, LoRALinear
from modules.attention import MultiHeadAttention
from modules.sparse.attention import SparseMultiHeadAttention

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]
ALL = {'self_attn.to_qkv', 'cross_attn.to_q', 'cross_attn.to_kv',
       'cross_attn2.to_q', 'cross_attn2.to_kv'}
LEGACY_ALL = ALL - {'cross_attn2.to_q', 'cross_attn2.to_kv'}
TARGETS = ('to_qkv', 'to_q', 'to_kv')


class AttentionBlock(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attn = attention(8, 2)
        self.cross_attn = attention(8, 2, type='cross')
        self.cross_attn2 = attention(8, 2, type='cross')

    def forward(self, x, a, b):
        return self.self_attn(x) + self.cross_attn(x, a) + self.cross_attn2(x, b)


def functions(source, names, namespace):
    path = ROOT / source
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class LoRAAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_all_wraps_every_ss_and_slat_attention_and_adapters_receive_gradients(self):
        for attention in (MultiHeadAttention, SparseMultiHeadAttention):
            model = AttentionBlock(attention).requires_grad_(False)
            x, a, b = torch.randn(2, 4, 8), torch.randn(2, 6, 8), torch.randn(2, 5, 8)
            if attention is MultiHeadAttention:
                baseline = model(x, a, b).detach()
            inserted = add_lora_to_attention(model, rank=2, alpha=4,
                                             attention_scope='all', target_modules=TARGETS)
            self.assertEqual(set(inserted), ALL)
            if attention is MultiHeadAttention:
                torch.testing.assert_close(model(x, a, b), baseline)
                model(x, a, b).square().mean().backward()
            else:
                # SLat kernels require CUDA; its real QKV projections run on CPU.
                sum(module(torch.randn(3, module.linear.in_features)).square().mean()
                    for module in model.modules() if isinstance(module, LoRALinear)).backward()
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    self.assertFalse(module.linear.weight.requires_grad)
                    self.assertTrue(module.lora_A.weight.requires_grad)
                    self.assertTrue(module.lora_B.weight.requires_grad)
                    self.assertGreater(module.lora_B.weight.grad.abs().sum().item(), 0)
            self.assertFalse(model.self_attn.to_out.weight.requires_grad)

    def test_cross_scope_preserves_both_cross_branches_without_self_attention(self):
        for attention in (MultiHeadAttention, SparseMultiHeadAttention):
            model = AttentionBlock(attention)
            inserted = add_lora_to_attention(model, target_modules=TARGETS, attention_scope='cross')
            self.assertEqual(set(inserted), ALL - {'self_attn.to_qkv'})
            self.assertIsInstance(model.self_attn.to_qkv, nn.Linear)

    def test_real_optimizer_group_contains_all_adapters_at_requested_rate(self):
        namespace = functions('train.py', {'get_flow_module', 'collect_param_groups'},
                              dict(torch=torch, Tuple=Tuple, List=List, Dict=Dict, Any=Any, Optional=Optional))
        for attention, name, rate in ((MultiHeadAttention, 'sparse_structure_flow', 1e-4),
                                      (SparseMultiHeadAttention, 'slat_flow', 5e-5)):
            model = nn.Module()
            flow = AttentionBlock(attention).requires_grad_(False)
            setattr(model, name, flow)
            add_lora_to_attention(flow, rank=2, alpha=4, target_modules=TARGETS)
            args = SimpleNamespace(use_lora=1, lr=rate, cond_lr=rate, flow_lr=rate, lora_lr=rate)
            groups, unused_cond_lr, unused_flow_lr = namespace['collect_param_groups'](model, args)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]['name'], 'lora')
            self.assertEqual(groups[0]['lr'], rate)
            self.assertEqual(len(groups[0]['params']), 10)
            self.assertEqual({id(p) for p in groups[0]['params']},
                             {id(p) for p in model.parameters() if p.requires_grad})

    def test_evaluator_strictly_loads_legacy_all_cross_and_complete_all_checkpoints(self):
        namespace = functions('eval_validation_latents.py', {'get_flow_module', 'maybe_insert_lora'},
                              {'add_lora_to_attention': add_lora_to_attention})
        args = {'use_lora': 1, 'lora_rank': 2, 'lora_alpha': 4, 'lora_dropout': 0.,
                'lora_target_modules': ','.join(TARGETS), 'lora_attention_scope': 'all'}
        for attention, name in ((MultiHeadAttention, 'sparse_structure_flow'),
                                 (SparseMultiHeadAttention, 'slat_flow')):
            for selected in (LEGACY_ALL, ALL - {'self_attn.to_qkv'}, ALL):
                original = nn.Module()
                setattr(original, name, AttentionBlock(attention))
                add_lora_to_attention(getattr(original, name), rank=2, alpha=4, module_names=selected)
                checkpoint = original.state_dict()
                restored = nn.Module()
                setattr(restored, name, AttentionBlock(attention))
                with redirect_stdout(io.StringIO()):
                    namespace['maybe_insert_lora'](restored, args, checkpoint)
                restored.load_state_dict(checkpoint, strict=True)
                actual = {path for path, module in getattr(restored, name).named_modules()
                          if isinstance(module, LoRALinear)}
                self.assertEqual(actual, selected)


if __name__ == '__main__':
    unittest.main()
