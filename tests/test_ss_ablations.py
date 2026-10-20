"""CPU numerical checks for fixed-alpha attention and budget-matched source swap."""
import io
import ast
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'xformers'
import torch
from torch import nn
torch.set_num_threads(2)
from data.morph_dataset import MorphingDistillDataset
from modules.transformer.modulated import ModulatedTransformerCrossBlock
from modules.sparse.basic import SparseTensor
from modules.sparse.transformer.modulated import ModulatedSparseTransformerCrossBlock


class FixedAlphaAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(15)
        self.block = ModulatedTransformerCrossBlock(
            8, 8, 2, separate_cond=True, separate_cond_gate='alpha', share_mod=True)
        self.x = torch.randn(3, 5, 8)
        self.c1 = torch.randn(3, 4, 8, requires_grad=True)
        self.c2 = torch.randn(3, 4, 8, requires_grad=True)
        self.alpha = torch.tensor([1., .25, 0.])
        # Zero residual self-attention/MLP modulation isolates the cross branches.
        self.mod = torch.zeros(3, 48)

    def test_two_real_attention_outputs_are_alpha_weighted_and_keep_endpoints(self):
        h1 = self.block.cross_attn(self.block.norm2(self.x), self.c1)
        h2 = self.block.cross_attn2(self.block.norm4(self.x), self.c2)
        alpha = self.alpha.view(-1, 1, 1)
        actual = self.block(self.x, self.mod, (self.c1, self.c2, self.alpha))
        torch.testing.assert_close(actual, self.x + alpha*h1 + (1-alpha)*h2)
        torch.testing.assert_close(actual[0], self.x[0]+h1[0])
        torch.testing.assert_close(actual[2], self.x[2]+h2[2])
        self.assertFalse(hasattr(self.block, 'alpha_gate'))
        self.assertTrue(hasattr(self.block, 'cross_attn2'))
        actual.square().mean().backward()
        for branch in (self.block.cross_attn, self.block.cross_attn2):
            self.assertGreater(branch.to_kv.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.c1.grad.abs().sum().item(), 0)
        self.assertGreater(self.c2.grad.abs().sum().item(), 0)

    def test_original_zero_initialized_token_gate_matches_fixed_alpha(self):
        token = ModulatedTransformerCrossBlock(8, 8, 2, separate_cond=True,
                                               separate_cond_gate='token', share_mod=True)
        token.load_state_dict(self.block.state_dict(), strict=False)
        context = (self.c1, self.c2, self.alpha)
        torch.testing.assert_close(token(self.x, self.mod, context), self.block(self.x, self.mod, context))
        self.assertTrue(hasattr(token, 'alpha_gate'))
        with torch.no_grad():
            token.alpha_gate[-1].bias.fill_(1.)
        changed = token(self.x, self.mod, context)
        fixed = self.block(self.x, self.mod, context)
        torch.testing.assert_close(changed[[0, 2]], fixed[[0, 2]])
        self.assertFalse(torch.allclose(changed[1], fixed[1]))

    def test_fixed_alpha_checkpoint_roundtrip_is_strict(self):
        buffer = io.BytesIO()
        torch.save(self.block.state_dict(), buffer)
        buffer.seek(0)
        other = ModulatedTransformerCrossBlock(8, 8, 2, separate_cond=True,
                                               separate_cond_gate='alpha', share_mod=True)
        other.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        context = (self.c1, self.c2, self.alpha)
        torch.testing.assert_close(other(self.x, self.mod, context), self.block(self.x, self.mod, context))

    def test_invalid_gate_is_rejected(self):
        with self.assertRaises(ValueError):
            ModulatedTransformerCrossBlock(8, 8, 2, separate_cond=True, separate_cond_gate='unknown')

    def test_evaluator_reads_checkpoint_metadata_and_strictly_loads_both_modes(self):
        # Exercise the real evaluator constructor/validation without allocating
        # the full image_large student. Real attention parameters are loaded.
        class SmallStudent(nn.Module):
            def __init__(self, separate_cond=True, separate_cond_gate='token'):
                super().__init__()
                self.block = ModulatedTransformerCrossBlock(
                    8,8,2,separate_cond=separate_cond,separate_cond_gate=separate_cond_gate)
        path = Path(__file__).resolve().parents[1]/'eval_validation_latents.py'
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='build_model')
        namespace = dict(inspect=inspect,torch=torch,MorphFlow=SmallStudent,MorphSLatFlow=SmallStudent,
                         checkpoint_args=lambda ckpt:ckpt['args'],unwrap_state_dict=lambda ckpt:ckpt['model'],
                         maybe_insert_lora=lambda model,args:None)
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        for mode in ('alpha','token'):
            original = SmallStudent(separate_cond_gate=mode)
            checkpoint = {'args':{'separate_cond':1,'separate_cond_gate':mode},'model':original.state_dict()}
            with redirect_stdout(io.StringIO()):
                restored = namespace['build_model'](checkpoint,'image_large','ss')
            self.assertEqual(restored.block.separate_cond_gate,mode)
            for name,value in original.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name],value)


class SparseContext(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, x, context):
        values = torch.empty_like(x.feats)
        for index, layout in enumerate(x.layout):
            values[layout] = context[index].mean(0) + self.offset
        return x.replace(values)


class SparseAlphaTests(unittest.TestCase):
    def test_alpha_is_broadcast_by_sparse_batch_with_unequal_row_counts(self):
        block = ModulatedSparseTransformerCrossBlock(8, 8, 2, separate_cond=True, separate_cond_gate='alpha')
        block.cross_attn = SparseContext(1.)
        block.cross_attn2 = SparseContext(-2.)
        coords = torch.tensor([[0,0,0,0], [0,0,0,1], [0,0,0,2], [1,0,0,0]], dtype=torch.int32)
        x = SparseTensor(torch.randn(4,8), coords)
        c1, c2 = torch.randn(2,3,8), torch.randn(2,3,8)
        alpha = torch.tensor([1., .2])
        actual = block._mix_separate_conditions(x,c1,c2,alpha)
        expected = torch.cat(((c1[0].mean(0)+1).expand(3,-1),
                              (.2*(c1[1].mean(0)+1)+.8*(c2[1].mean(0)-2)).view(1,-1)))
        torch.testing.assert_close(actual.feats,expected)
        torch.testing.assert_close(actual.coords,coords)
        self.assertEqual(actual.layout,x.layout)
        self.assertFalse(hasattr(block,'alpha_gate'))


class SourceSwapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root/'metadata_train.json').write_text(json.dumps([{'target':str(i)} for i in range(4)]))

    def dataset(self, augment=False):
        return MorphingDistillDataset(str(self.root),split='train',skip_missing=False,
                                      augment_source_swap=augment,verbose=False)

    def load(self, entry, swap_sources=False):
        sample = {'src1_name':'one','src2_name':'two','alpha':torch.tensor(.3),
                  'target_name':entry['target'],'target_feats':torch.tensor([99.])}
        for suffix in ('feats','coords','ss_latent','occupancy','image'):
            sample['src1_'+suffix] = torch.tensor([1.])
            sample['src2_'+suffix] = torch.tensor([2.])
        return MorphingDistillDataset._swap_sources(sample) if swap_sources else sample

    def test_historical_default_still_doubles_and_reverses_every_source_field(self):
        ds = self.dataset(augment=True)
        self.assertEqual(len(ds),8)
        with patch.object(ds,'_load_entry',side_effect=self.load):
            original, swapped = ds[0], ds[4]
        self.assertEqual(swapped['src1_name'],'two')
        self.assertEqual(swapped['src2_name'],'one')
        self.assertAlmostEqual(swapped['alpha'].item(),.7)
        for suffix in ('feats','coords','ss_latent','occupancy','image'):
            torch.testing.assert_close(original['src1_'+suffix],swapped['src2_'+suffix])
            torch.testing.assert_close(original['src2_'+suffix],swapped['src1_'+suffix])
        torch.testing.assert_close(original['target_feats'],swapped['target_feats'])
        self.assertEqual(original['target_name'],swapped['target_name'])

    def test_disabled_swap_keeps_length_orientation_and_rng_state(self):
        ds = self.dataset()
        self.assertEqual(len(ds),4)
        before = torch.get_rng_state().clone()
        with patch.object(ds,'_load_entry',side_effect=self.load):
            sample = ds[0]
        self.assertEqual(sample['src1_name'],'one')
        torch.testing.assert_close(torch.get_rng_state(),before)


if __name__ == '__main__':
    unittest.main()
