"""Endpoint weighting, sequence discovery and exported curve regression tests."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import eval_perceptual_sequence as metric


class WeightedEndpointTests(unittest.TestCase):
    def test_endpoints_and_middle_use_model_alpha_convention(self):
        rows, values = metric.weighted_endpoint_curve([1, .5, 0], [[.2, .4, .8]], [[.9, .6, .3]])
        np.testing.assert_allclose(values, [[.2, .5, .3]])
        self.assertEqual(rows[0]["src2_weighted_mean"], 0)
        self.assertEqual(rows[-1]["src1_weighted_mean"], 0)

    def test_swapping_sources_and_alpha_preserves_score(self):
        alpha = np.array([.1, .4, .85])
        d1 = np.array([[.8, .6, .2], [.9, .7, .3]])
        d2 = np.array([[.1, .3, .7], [.2, .4, .8]])
        rows, score = metric.weighted_endpoint_curve(alpha, d1, d2)
        _, swapped = metric.weighted_endpoint_curve(1-alpha, d2, d1)
        np.testing.assert_allclose(score, swapped)
        self.assertAlmostEqual(rows[1]["weighted_lpips_mean"], score[:, 1].mean())
        self.assertAlmostEqual(rows[1]["weighted_lpips_std_views"], score[:, 1].std())

    def test_invalid_alphas_shapes_and_scores_fail(self):
        for a in ([0, 0], [0, float("nan")], [-.1, 1], [0, 1.1]):
            with self.subTest(a=a), self.assertRaises(ValueError):
                metric.weighted_endpoint_curve(a, [[1, 2]], [[2, 1]])
        with self.assertRaises(ValueError):
            metric.weighted_endpoint_curve([0, 1], [[1, 2]], [[1]])
        with self.assertRaises(ValueError):
            metric.weighted_endpoint_curve([0, 1], [[1, float("inf")]], [[2, 1]])

    def test_grid_discovery_references_and_sorting(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("one.glb", "two.glb"):
                (root/name).touch()
            (root/"run.json").write_text(json.dumps({"config": {
                "src1": str(root/"one.glb"), "src2": str(root/"two.glb")}}))
            for cfg in ("cfg_ss_2_slat_2", "cfg_ss_3_slat_2p5"):
                for alpha in (0, .5, 1):
                    folder=root/cfg/str("alpha_"+str(alpha).replace(".", "p"))
                    folder.mkdir(parents=True); (folder/"prediction.glb").touch()
                    (folder/"result.json").write_text(json.dumps({"model_alpha": alpha}))
            groups = metric.discover_sequence_dirs(root, "pair_*")
            self.assertEqual(len(groups), 2)
            one, two, frames, _ = metric.discover_frames(groups[0])
            self.assertEqual((one, two), (root/"one.glb", root/"two.glb"))
            self.assertEqual([frame.alpha for frame in frames], [1, .5, 0])
            self.assertEqual(len({frame.key for frame in frames}), 3)

    def test_moved_sequence_and_duplicate_alphas(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for role in ("src1", "src2"):
                (root/(role+".glb")).touch()
            records=[]
            for a in (0, .5, 1):
                folder=root/("alpha_"+str(a).replace(".", "p"))
                folder.mkdir(); (folder/"pred_final.glb").touch()
                records.append({"kind":"prediction", "alpha":a,
                                "path":"/old/machine/"+folder.name+"/pred_final.glb"})
            (root/"sequence.json").write_text(json.dumps(records))
            _, _, frames, _ = metric.discover_frames(root)
            self.assertEqual(len(frames), 3)
            records[-1]["alpha"] = .5
            (root/"sequence.json").write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, "same alpha"):
                metric.discover_frames(root)

    def test_exported_curve_matches_lpips_pairs_and_nonuniform_auc(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            frames=[metric.Frame(i,a,root/f"{i}.glb",f"frame{i}") for i,a in enumerate([1,.2,0])]
            def distance(_model,pairs,_device,_batch):
                # Explicit source distances; adjacency values are irrelevant to this test.
                return [(.4 if b.parent.name == 'src1' else .8) for a,b in pairs]
            args=SimpleNamespace(num_views=2,device="cpu",lpips_net="vgg",lpips_batch_size=8)
            with patch.object(metric,"image_path",side_effect=lambda out,key,v:out/key/f"{v}.png"), \
                 patch.object(metric,"batched_lpips",side_effect=distance):
                result=metric.compute_metrics(args,root,frames,model=object(),device="cpu")
            rows=result['weighted_endpoint_curve']
            np.testing.assert_allclose([r['weighted_lpips_mean'] for r in rows],[.4,.72,.8])
            self.assertAlmostEqual(result['aggregate_across_views']['weighted_endpoint_lpips_auc']['mean'],.6)
            for name in ['weighted_endpoint_lpips.csv','weighted_endpoint_lpips_per_view.csv',
                         'weighted_endpoint_lpips.png','weighted_endpoint_lpips.pdf','metrics.json']:
                self.assertGreater((root/name).stat().st_size,0)


if __name__ == '__main__':
    unittest.main()
