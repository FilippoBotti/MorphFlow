"""CPU checks for budget-matched, shared Interp3D datasets with 8/2 alpha."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from data.dataset_scripts import generate_interp3d_v4 as base
from data.dataset_scripts import generate_interp3d_v4_multistep as joint
from slurm import audit_interp3d_v4 as audit
from slurm import monitor_interp3d_v4 as monitor


def pair_key(pair):
    return pair['split'], pair['src_1'], pair['src_2']


def target_counts(pairs, split):
    values = [a for p in pairs if p['split'] == split for a in p['alphas']]
    return len(values), values.count(0.0), values.count(1.0)


class MultiStepDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='interp3d_multistep_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.v3 = self.root / 'v3'
        self.images = self.root / 'images'
        self.out = self.root / 'dataset'
        self.v3.mkdir()
        self.images.mkdir()

    def inputs(self, sizes):
        splits = {}
        for split, size in zip(base.SPLITS, sizes):
            splits[split] = ['%s_%03d' % (split, i) for i in range(size)]
            for name in splits[split]:
                (self.images / (name + '.png')).write_bytes(b'planner-only image')
        base.save_json(self.v3 / 'split_assets.json', splits)
        return splits

    def args(self, samples=(80, 8, 8), **overrides):
        options = dict(v3_root=str(self.v3), images_root=str(self.images),
                       out=str(self.out), train_samples=samples[0],
                       val_samples=samples[1], test_samples=samples[2],
                       endpoint_fraction=0.02, seed=42,
                       alpha_min=0.001, alpha_max=0.999)
        options.update(overrides)
        return SimpleNamespace(**options)

    def plan(self, args):
        with redirect_stdout(io.StringIO()):
            joint.command_plan(args)
        master = base.read_json(self.out / 'plan.json')
        views = {k: base.read_json(self.view(k) / 'plan.json') for k in (8, 2)}
        return master, views

    def view(self, k):
        return self.out / ('morphing_dataset_v4_steps%d' % k)

    def assert_equal_bins(self, pair, k):
        values = pair['alphas']
        self.assertEqual(len(values), k)
        self.assertEqual(len(set(values)), k)
        self.assertEqual(values, sorted(values, reverse=True))
        width = (0.999 - 0.001) / k
        for j, alpha in enumerate(reversed(values)):
            if alpha == 0.0:
                self.assertEqual(j, 0)
            elif alpha == 1.0:
                self.assertEqual(j, k - 1)
            else:
                self.assertGreaterEqual(alpha, 0.001 + j * width - 0.000001)
                self.assertLessEqual(alpha, 0.001 + (j + 1) * width + 0.000001)

    def test_production_budgets_bins_endpoints_and_hybrid_teacher_reuse(self):
        assignments = self.inputs((225, 46, 46))
        master, views = self.plan(self.args(samples=(50000, 2000, 2000)))
        self.assertIsNone(master['k'])
        self.assertTrue(master['variable_alphas_per_pair'])
        self.assertEqual(master['num_pairs'], 27000)
        self.assertEqual(master['num_targets'], 94500)
        self.assertEqual(sum(len(p['alphas']) for p in master['pairs']), 94500)
        expected_quotas = {8: {'train': 6250, 'val': 250, 'test': 250},
                           2: {'train': 25000, 'val': 1000, 'test': 1000}}
        master_by_pair = {pair_key(p): p for p in master['pairs']}
        self.assertEqual(len(master_by_pair), master['num_pairs'])
        by_view = {k: {pair_key(p): p for p in view['pairs']}
                   for k, view in views.items()}
        self.assertTrue(set(by_view[8]).issubset(by_view[2]))
        for k, view in views.items():
            self.assertEqual(view['k'], k)
            self.assertEqual(view['num_targets'], 54000)
            self.assertEqual(view['pair_quotas'], expected_quotas[k])
            self.assertEqual(view['train_virtual_after_swap'], 100000)
            for split, size in zip(base.SPLITS, (50000, 2000, 2000)):
                self.assertEqual(target_counts(view['pairs'], split),
                                 (size, size // 100, size // 100))
                self.assertEqual(len([p for p in view['pairs'] if p['split'] == split]),
                                 expected_quotas[k][split])
            for pair in view['pairs']:
                self.assert_equal_bins(pair, k)
                teacher = master_by_pair[pair_key(pair)]
                self.assertEqual(pair['seed'], teacher['seed'])
                self.assertEqual(pair['teacher_alphas'], teacher['alphas'])
                self.assertTrue(set(pair['alphas']).issubset(teacher['alphas']))
                self.assertIn(pair['src_1'], assignments[pair['split']])
                self.assertIn(pair['src_2'], assignments[pair['split']])
        for key, teacher in master_by_pair.items():
            shared = key in by_view[8]
            self.assertEqual(teacher['shared_with_steps8'], shared)
            self.assertEqual(len(teacher['alphas']), 8 if shared else 2)
            self.assertEqual(teacher['selected_alphas_steps2'], by_view[2][key]['alphas'])
            if shared:
                self.assertEqual(teacher['alphas'], by_view[8][key]['alphas'])
            else:
                self.assertEqual(teacher['alphas'], by_view[2][key]['alphas'])
        shared2 = [p for p in views[2]['pairs'] if pair_key(p) in by_view[8]]
        extra2 = [p for p in views[2]['pairs'] if pair_key(p) not in by_view[8]]
        self.assertEqual(target_counts(shared2, 'train'), (12500, 125, 125))
        self.assertEqual(target_counts(extra2, 'train'), (37500, 375, 375))

    def test_resume_preserves_plans_and_rejects_changed_seed_or_budget(self):
        self.inputs((24, 10, 10))
        args = self.args()
        self.plan(args)
        paths = [self.out / 'plan.json'] + [self.view(k) / 'plan.json' for k in (8, 2)]
        before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
        self.plan(args)
        self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])
        for changed in (self.args(seed=43), self.args(train_samples=88)):
            with self.assertRaises((RuntimeError, ValueError)), redirect_stdout(io.StringIO()):
                joint.command_plan(changed)
        self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])

    def test_invalid_budget_and_asset_leakage_are_rejected(self):
        assignments = self.inputs((24, 10, 10))
        for samples in ((81, 8, 8), (-8, 8, 8), (0, 0, 0)):
            with self.assertRaises((RuntimeError, ValueError)), redirect_stdout(io.StringIO()):
                joint.command_plan(self.args(samples=samples))
        assignments['val'].append(assignments['train'][0])
        base.save_json(self.v3 / 'split_assets.json', assignments)
        with self.assertRaisesRegex(RuntimeError, '[Ll]eakage'), redirect_stdout(io.StringIO()):
            joint.command_plan(self.args())

    def test_finalize_requires_all_targets_then_shares_bytes_and_teacher_provenance(self):
        self.inputs((24, 10, 10))
        args = self.args()
        master, views = self.plan(args)
        with self.assertRaises(RuntimeError), redirect_stdout(io.StringIO()):
            joint.command_finalize(SimpleNamespace(out=str(self.out)))
        for asset in master['assets']:
            self.fake_ready(self.out / 'assets' / asset['name'])
        for pair in master['pairs']:
            for alpha in pair['alphas']:
                if alpha not in (0.0, 1.0):
                    self.fake_ready(base.target_path(self.out, pair['src_1'], pair['src_2'], alpha))
        with redirect_stdout(io.StringIO()):
            joint.command_finalize(SimpleNamespace(out=str(self.out)))
        master_by_pair = {pair_key(p): p for p in master['pairs']}
        metadata = {}
        for k in (8, 2):
            view = self.view(k)
            for name in ('assets', 'targets'):
                self.assertTrue((view / name).is_symlink())
                self.assertFalse(os.path.isabs(os.readlink(str(view / name))))
                self.assertEqual((view / name).resolve(), (self.out / name).resolve())
            self.assertEqual(base.read_json(view / 'missing_targets.json'), [])
            metadata[k] = base.read_json(view / 'metadata.json')
            self.assertEqual(len(metadata[k]), 96)
            self.assertEqual(base.read_json(view / 'splits.json')['metadata_counts'],
                             {'train': 80, 'val': 8, 'test': 8})
            for entry in metadata[k]:
                teacher = master_by_pair[pair_key(entry)]
                self.assertEqual(entry['sequence_alphas'], teacher['alphas'])
                self.assertEqual(entry['sequence_intermediates'],
                                 sum(0.0 < a < 1.0 for a in teacher['alphas']))
                if 0.0 < entry['alpha'] < 1.0:
                    self.assertEqual(entry['alpha_sequence_index'],
                                     teacher['alphas'].index(entry['alpha']))
                for field in ('src1_ss_latent', 'src2_ss_latent', 'target_ss_latent'):
                    self.assertTrue((view / entry[field]).is_file())
        samples8 = {(pair_key(e), e['alpha']): e for e in metadata[8]}
        reused = [e for e in metadata[2] if (pair_key(e), e['alpha']) in samples8]
        self.assertEqual(len(reused), 24)
        for entry in reused:
            left = self.view(8) / entry['target_ss_latent']
            right = self.view(2) / entry['target_ss_latent']
            self.assertEqual(left.resolve(), right.resolve())
            self.assertEqual(left.stat().st_ino, right.stat().st_ino)
        before = {k: (self.view(k) / 'metadata.json').read_bytes() for k in (8, 2)}
        with redirect_stdout(io.StringIO()):
            joint.command_finalize(SimpleNamespace(out=str(self.out)))
        self.assertEqual(before, {k: (self.view(k) / 'metadata.json').read_bytes() for k in (8, 2)})

    def fake_ready(self, directory, archives=False):
        directory.mkdir(parents=True, exist_ok=True)
        for name in base.REQUIRED:
            if archives:
                with zipfile.ZipFile(str(directory / name), 'w') as archive:
                    archive.writestr('fixture/data', b'CPU fixture for archive CRC checks')
            else:
                (directory / name).write_bytes(b'CPU fixture; tensor payload is not read')
        base.save_json(directory / 'ready.json', {'test_fixture': True})

    def test_generic_audit_accepts_both_views_detects_corruption_and_monitor_shows_mixed_plan(self):
        self.inputs((24, 10, 10))
        master, unused_views = self.plan(self.args())
        for asset in master['assets']:
            self.fake_ready(self.out / 'assets' / asset['name'], archives=True)
        for pair in master['pairs']:
            for alpha in pair['alphas']:
                if alpha not in (0.0, 1.0):
                    self.fake_ready(base.target_path(self.out, pair['src_1'], pair['src_2'], alpha),
                                    archives=True)
        with redirect_stdout(io.StringIO()):
            joint.command_finalize(SimpleNamespace(out=str(self.out)))
        state = {'phase': 'complete', 'status': 'complete',
                 'job_id': 'cpu-test', 'run_id': 'cpu-test'}
        base.save_json(self.out / 'run_state.json', state)
        for k in (8, 2):
            base.save_json(self.view(k) / 'run_state.json', state)
            with patch('sys.argv', ['audit_interp3d_v4.py', '--out', str(self.view(k))]), \
                    redirect_stdout(io.StringIO()) as output:
                result = audit.main()
            self.assertEqual(result, 0, output.getvalue())
            self.assertIn('RESULT: PASS', output.getvalue())
        status = monitor.status_data(str(self.out))
        self.assertEqual(status['plan']['alphas_per_pair'], [2, 8])
        self.assertTrue(status['plan']['variable_alphas_per_pair'])
        with redirect_stdout(io.StringIO()) as output:
            monitor.show(status)
        self.assertIn('2/8 alphas per pair', output.getvalue())
        for k in (8, 2):
            self.assertIn('DATASET morphing_dataset_v4_steps%d:' % k, output.getvalue())
        self.assertIn('virtual train=160', output.getvalue())
        self.assertNotIn('None', output.getvalue())
        rows = base.read_json(self.view(2) / 'metadata.json')
        # The audit always samples the lexicographically first target's SS ZIP.
        first_target = sorted({str(self.view(2) / e['target_dir']) for e in rows})[0]
        (Path(first_target) / 'ss_latent.pt').write_bytes(b'corrupted ZIP archive')
        with patch('sys.argv', ['audit_interp3d_v4.py', '--out', str(self.view(2))]), \
                redirect_stdout(io.StringIO()) as output:
            result = audit.main()
        self.assertEqual(result, 1)
        self.assertIn('Corrupted PyTorch archive', output.getvalue())
        self.assertIn('RESULT: FAIL', output.getvalue())

    def test_legacy_four_step_schedule_is_unchanged(self):
        schedule = base.assign_pair_alpha_sets(100, 4, 17, 'test', 0.02, 0.001, 0.999)
        digest = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode('utf8')).hexdigest()
        self.assertEqual(digest, 'f4459e2aea3dcdfe12598a035da100ae3bf4f09417f0d32a661b301f18f8a897')

    def test_mixed_worker_progress_counts_targets_and_keeps_legacy_defaults(self):
        with redirect_stdout(io.StringIO()):
            reporter = base.Reporter(self.out, 'pairs', 0, 2, target_total=10)
            try:
                reporter.complete('train', 'generated', 'interior', targets_generated=7,
                                  target_count=8)
                reporter.complete('val', 'reused', 'interior', target_count=2)
                reporter.finish()
            finally:
                reporter.close()
        state = base.read_json(reporter.path)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual((state['completed'], state['total']), (2, 2))
        self.assertEqual((state['target_completed'], state['target_total']), (10, 10))
        self.assertEqual((state['target_generated'], state['target_reused']), (7, 3))
        self.assertEqual(state['per_split_targets'], {'train': 8, 'val': 2, 'test': 0})
        with redirect_stdout(io.StringIO()):
            legacy = base.Reporter(self.out, 'pairs', 1, 2, targets_per_pair=4)
            try:
                legacy.complete('train', 'generated', 'interior', targets_generated=4)
                legacy.complete('train', 'reused', 'interior')
                legacy.finish()
            finally:
                legacy.close()
        state = base.read_json(legacy.path)
        self.assertEqual((state['target_completed'], state['target_total']), (8, 8))
        self.assertEqual((state['target_generated'], state['target_reused']), (4, 4))


if __name__ == '__main__':
    unittest.main()
