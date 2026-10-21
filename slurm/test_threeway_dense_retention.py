import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('dense_retention', Path(__file__).with_name('threeway_dense_retention.py'))
retention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retention)


def fixture(root):
    pair = 'pair_0000_a_to_b'
    plan = dict(metric_mode='weighted-endpoints-only',k_intermediate=48,num_pairs=1,pairs=[dict(pair_dir=pair)])
    (root/'plan.json').write_text(json.dumps(plan))
    output = root/'metrics/morphflow'/pair
    output.mkdir(parents=True)
    result = dict(metric_mode='weighted-endpoints-only',num_generated_frames=48,num_sequence_frames=50,
                  num_views=64,weighted_endpoint_includes_reference_endpoints=True,
                  weighted_endpoint_curve=[{} for _ in range(50)],
                  aggregate_across_views={m:dict(mean=.2) for m in retention.METRICS})
    (output/'metrics.json').write_text(json.dumps(result))
    batch = dict(metric_mode='weighted-endpoints-only',num_generated_frames=48,num_sequence_frames=50,
                 num_views_per_pair=64,num_pairs=1,missing_pairs=[],pairs=[dict(sequence=pair)])
    (output.parent/'metrics_batch.json').write_text(json.dumps(batch))
    generated = root/'morphflow'/pair/'alpha_0p5'
    generated.mkdir(parents=True)
    (generated/'mesh_raw.pt').write_bytes(b'raw')
    (generated/'mesh_raw.appearance.json').write_text('{}')
    (generated/'keep.json').write_text('{}')
    (output/'image.png').write_bytes(b'render')
    return output,generated


class DenseRetentionTests(unittest.TestCase):
    def test_streamed_pair_retirement_does_not_require_method_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output,generated=fixture(root)
            (output.parent/'metrics_batch.json').unlink()
            pair='pair_0000_a_to_b'
            retention.release(root,'morphflow',pair)
            self.assertTrue(retention.retired(root,'morphflow',pair))
            self.assertFalse(retention.retired(root,'morphflow'))
            self.assertFalse((generated/'mesh_raw.pt').exists())
            with self.assertRaises(ValueError):
                retention.release(root,'morphflow','pair_unplanned')

    def test_cleanup_keeps_renders_metrics_and_resumes_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output,generated=fixture(root)
            self.assertFalse(retention.retired(root,'morphflow'))
            retention.release(root,'morphflow')
            self.assertTrue(retention.retired(root,'morphflow'))
            self.assertFalse((generated/'mesh_raw.pt').exists())
            self.assertTrue((generated/'keep.json').exists())
            self.assertTrue((output/'image.png').exists())
            retention.release(root,'morphflow')
            # A changed metric invalidates the durable retirement marker.
            result=json.loads((output/'metrics.json').read_text())
            result['aggregate_across_views'][retention.METRICS[0]]['mean']=.3
            (output/'metrics.json').write_text(json.dumps(result))
            self.assertFalse(retention.retired(root,'morphflow'))

    def test_seven_state_or_incomplete_metrics_cannot_delete_raw(self):
        for defect in ['seven','incomplete','nonfinite']:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);output,generated=fixture(root)
                if defect=='seven':
                    plan=json.loads((root/'plan.json').read_text());plan['k_intermediate']=5
                    (root/'plan.json').write_text(json.dumps(plan))
                elif defect=='incomplete':
                    (output/'metrics.json').unlink()
                else:
                    result=json.loads((output/'metrics.json').read_text())
                    result['aggregate_across_views'][retention.METRICS[0]]['mean']=float('nan')
                    (output/'metrics.json').write_text(json.dumps(result))
                with self.assertRaises((ValueError,FileNotFoundError)):
                    retention.release(root,'morphflow')
                self.assertTrue((generated/'mesh_raw.pt').exists())


if __name__ == '__main__':
    unittest.main()
