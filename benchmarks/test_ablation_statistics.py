"""CPU checks for statistical definitions and interrupted-batch recovery."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,str(Path(__file__).parent/file))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


stream=module('stream_statistics_test','stream_ablation_statistics.py')
report=module('report_statistics_test','ablation_statistics_report.py')
ALPHAS=[.65,.57,.5,.43,.35]


def result(pair):
    return dict(num_generated_frames=5,num_views=64,lpips_backbone='vgg',render_backend='gaussian',
                full_path_includes_reference_endpoints=True,alphas=ALPHAS,sequence_name=pair['pair_dir'],
                evaluation_signature='synthetic',fid_protocol=dict(
                    views_per_state=1000,selected_indices=[0,2],
                    selected_alphas=[ALPHAS[0],ALPHAS[2]],
                    reference_samples=2000,generated_samples=2000,
                    selection='seeded_random_without_replacement',selection_seed=42,
                    aggregation_modes=['concatenated']),
                aggregate_across_views={k:dict(mean=v) for k,v in dict(fid_endpoint_manifold=90.,pdv_full=.001,
                  ppl_full_normalized_reference_endpoints=3.,reference_endpoint_lpips=.25).items()},
                weighted_endpoint_curve=[dict(alpha=a,lpips_src1_mean=(1-a)*.25,lpips_src2_mean=a*.25) for a in ALPHAS])


def fake_metrics(cell,pair):
    folder=cell/'metrics/morphflow'/pair['pair_dir'];folder.mkdir(parents=True,exist_ok=True)
    (folder/'metrics.json').write_text(json.dumps(result(pair)))
    (folder/'full_per_view.csv').write_text('view,pdv\n0,0.001\n')
    for key in ['src1']+[('a%.6f'%a).replace('.','p') for a in ALPHAS]+['src2']:
        path=folder/'images'/key/'025.png';path.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(512,512),'white').save(path)


def fake_worker_run(root,pair_ids,batch_shards):
    pairs=[dict(pair_id=i,pair_dir='pair_%04d_a_to_b'%i,src1='a'+str(i),src2='b'+str(i),
                pair_seed=42+i*100000,confirmation=True,pilot=False,source_component=i) for i in pair_ids]
    stream.write(root/'plan.json',dict(pairs=pairs,alphas=ALPHAS))
    checkpoint=root/'checkpoint.pt';checkpoint.write_bytes(b'frozen')
    stat=checkpoint.stat()
    stream.write(root/'run_manifest.json',dict(variants=['baseline'],cfg_ss=5.,cfg_slat=1.5,
                 batch_size=2,batch_shards=batch_shards,checkpoints=dict(baseline={
                     flow:dict(path=str(checkpoint),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
                     for flow in ['ss','slat']})))
    return pairs


class StatisticsTests(unittest.TestCase):
    def test_concatenated_fid_metadata_without_per_state_scores(self):
        p=dict(pair_dir='pair_0000_a_to_b')
        value=result(p)
        self.assertNotIn('selected_states',value['fid_protocol'])
        scores,_=stream.scores(value,dict(alphas=ALPHAS))
        self.assertEqual(scores['FID'],90.)
        # Per-state diagnostics can be present in older dual-mode archives,
        # but frame identities always come from selected_indices/alphas.
        value['fid_protocol'].update(
            aggregation_modes=['concatenated','mean-per-state'],
            selected_states=[dict(frame_index=0,fid=100.),dict(frame_index=2,fid=110.)])
        self.assertEqual(stream.scores(value,dict(alphas=ALPHAS))[0],scores)

    def test_invalid_selected_fid_frames_are_rejected(self):
        changes=[dict(selected_indices=[0,0]),dict(selected_indices=[0,5]),
                 dict(selected_indices=[False,2]),dict(selected_indices=[0]),
                 dict(selected_alphas=[ALPHAS[0]]),dict(selected_alphas=[ALPHAS[0],.9]),
                 dict(selected_alphas=[float('nan'),ALPHAS[2]]),
                 dict(reference_samples=1000),dict(generated_samples=1000),
                 dict(aggregation_modes=['mean-per-state'])]
        for fields in changes:
            with self.subTest(fields=fields):
                value=result(dict(pair_dir='pair_0000_a_to_b'))
                value['fid_protocol'].update(fields)
                with self.assertRaises((AssertionError,ValueError)):
                    stream.scores(value,dict(alphas=ALPHAS))
        for missing in ['selected_indices','selected_alphas']:
            with self.subTest(missing=missing):
                value=result(dict(pair_dir='pair_0000_a_to_b'))
                del value['fid_protocol'][missing]
                with self.assertRaises(ValueError):
                    stream.scores(value,dict(alphas=ALPHAS))

    def test_linear_profile_is_zero_and_invalid_values_rejected(self):
        p=dict(pair_dir='pair_0000_a_to_b')
        scores,points=stream.scores(result(p),dict(alphas=ALPHAS))
        self.assertAlmostEqual(scores['EPRE'],0.,places=15)
        self.assertEqual(len(points),5)
        value=result(p);value['aggregate_across_views']['fid_endpoint_manifold']['mean']=float('nan')
        with self.assertRaises(AssertionError):stream.scores(value,dict(alphas=ALPHAS))

    def test_holm_and_seeded_paired_permutations(self):
        np.testing.assert_allclose(report.holm([.01,.04,.03]),[.03,.06,.06])
        values=np.stack([np.ones(20),np.array([1.,-1.]*10)],axis=1)
        a=report.paired_permutation_p(values,9999,42)
        b=report.paired_permutation_p(values,9999,42)
        np.testing.assert_array_equal(a,b)
        self.assertGreaterEqual(a[0],1/10000)
        self.assertLess(a[0],.01)
        self.assertEqual(a[1],1.)

    def test_cluster_bootstrap_keeps_constant_paired_differences(self):
        values=np.arange(24,dtype=float).reshape(6,4)
        values=np.stack([values,values+7.],axis=1)
        means=report.bootstrap_means(values,300,42,clusters=[0,0,0,1,2,2])
        np.testing.assert_allclose(means[:,1]-means[:,0],7.)

    def test_resume_after_partial_commit_and_safe_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pairs=[dict(pair_id=i,pair_dir='pair_%04d_a_to_b'%i,src1='a'+str(i),src2='b'+str(i),
                   pair_seed=42+i*100000,confirmation=True,pilot=False,source_component=i) for i in range(2)]
            stream.write(root/'plan.json',dict(pairs=pairs,alphas=ALPHAS))
            fake_checkpoint=root/'checkpoint.pt';fake_checkpoint.write_bytes(b'frozen')
            st=fake_checkpoint.stat()
            stream.write(root/'run_manifest.json',dict(variants=['baseline'],cfg_ss=5.,cfg_slat=1.5,batch_size=2,
                         checkpoints=dict(baseline={f:dict(path=str(fake_checkpoint),bytes=st.st_size,mtime_ns=st.st_mtime_ns) for f in ['ss','slat']})))
            seen=[]
            def phase(root,variant,label,environment,repository,args):
                if label.endswith('_generation'):
                    cell=Path(args[args.index('--run-root')+1]);plan=json.loads((cell/'plan.json').read_text())
                    for p in plan['pairs']:(cell/'morphflow'/p['pair_dir']).mkdir(parents=True,exist_ok=True)
                    actual={p.name for p in (cell/'morphflow').iterdir() if p.is_dir()}
                    self.assertEqual(actual,{p['pair_dir'] for p in plan['pairs']})
                    seen.append(len(plan['pairs']))
                if label.endswith('_metrics'):
                    cell=Path(args[args.index('--run-dir')+1]).parent
                    for p in json.loads((cell/'plan.json').read_text())['pairs']:fake_metrics(cell,p)
            original=stream.commit_pair
            def interrupt(root,variant,pair,cell,plan):
                if pair['pair_id']==1:raise InterruptedError('injected interruption')
                return original(root,variant,pair,cell,plan)
            args=type('Args',(),dict(run_root=str(root),variant='baseline'))()
            env=dict(ORIGINAL_ENV='/unused',METRIC_ENV='/unused',DATA_ROOT_HOST='/unused',SOURCE_IMAGES_ROOT_HOST='/unused')
            with patch.dict(os.environ,env),patch.object(stream,'phase',phase):
                with patch.object(stream,'commit_pair',interrupt):
                    with self.assertRaises(InterruptedError):stream.worker(args)
                self.assertTrue(stream.committed(root/'variants/baseline',pairs[0]))
                stream.worker(args)
            self.assertEqual(seen,[2,1])
            self.assertTrue((root/'variants/baseline/COMPLETE.json').is_file())
            self.assertFalse((root/'work/baseline/batch_0000').exists())
            outside=root/'do_not_delete';outside.mkdir();(outside/'asset').write_bytes(b'keep')
            with self.assertRaises(ValueError):stream.release_work(root,'baseline',outside,pairs)
            self.assertTrue((outside/'asset').is_file())
            marker=json.loads((root/'variants/baseline/pairs'/(pairs[0]['pair_dir']+'.json')).read_text())
            file=root/'variants/baseline'/marker['retained_files'][0]['path'];file.write_bytes(b'tampered')
            with self.assertRaises(RuntimeError):stream.committed(root/'variants/baseline',pairs[0])

    def test_parallel_batch_shards_partition_plan_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pairs=fake_worker_run(root,[41,8,92,5,104,33,2,78,67,6,55],3)
            run=root/'variants/baseline'
            expected={shard:[(batch,pairs[index:index+2]) for batch,index in enumerate(range(0,len(pairs),2))
                             if batch%3==shard] for shard in range(3)}
            observed={shard:[] for shard in range(3)}
            active_shard=0
            def phase(root,variant,label,environment,repository,args):
                if label.endswith('_generation'):
                    cell=Path(args[args.index('--run-root')+1])
                    batch_pairs=json.loads((cell/'plan.json').read_text())['pairs']
                    observed[active_shard].append((cell.name,[p['pair_id'] for p in batch_pairs]))
                    for pair in batch_pairs:(cell/'morphflow'/pair['pair_dir']).mkdir(parents=True,exist_ok=True)
                if label.endswith('_metrics'):
                    cell=Path(args[args.index('--run-dir')+1]).parent
                    for pair in json.loads((cell/'plan.json').read_text())['pairs']:fake_metrics(cell,pair)
            args=lambda shard:type('Args',(),dict(run_root=str(root),variant='baseline',
                                                batch_shard_index=shard,batch_shard_count=3))()
            env=dict(ORIGINAL_ENV='/unused',METRIC_ENV='/unused',DATA_ROOT_HOST='/unused',SOURCE_IMAGES_ROOT_HOST='/unused')
            committed_markers={}
            with patch.dict(os.environ,env),patch.object(stream,'phase',phase),patch.object(
                    stream.shutil,'disk_usage',return_value=type('Disk',(),dict(free=1024**4))()):
                for active_shard in range(3):
                    stream.worker(args(active_shard))
                    wanted=[('batch_%04d'%batch,[p['pair_id'] for p in batch_pairs])
                            for batch,batch_pairs in expected[active_shard]]
                    self.assertEqual(observed[active_shard],wanted)
                    for name,value in committed_markers.items():
                        self.assertEqual((run/'pairs'/name).read_bytes(),value)
                    completed_pairs=[p for shard in range(active_shard+1) for _,batch_pairs in expected[shard] for p in batch_pairs]
                    self.assertEqual({p.stem for p in (run/'pairs').glob('*.json')},{p['pair_dir'] for p in completed_pairs})
                    self.assertTrue(all(stream.committed(run,p) for p in completed_pairs))
                    status=json.loads((run/'status.json').read_text())
                    self.assertEqual((status['completed'],status['expected']),(len(completed_pairs),len(pairs)))
                    tag='shard_%03d_of_003'%active_shard
                    self.assertTrue((run/('.worker.'+tag+'.lock')).is_file())
                    self.assertTrue((run/('COMPLETE_'+tag+'.json')).is_file())
                    status=json.loads((run/('status_'+tag+'.json')).read_text())
                    shard_pair_count=sum(len(batch_pairs) for _,batch_pairs in expected[active_shard])
                    self.assertEqual((status['completed'],status['expected']),(shard_pair_count,shard_pair_count))
                    self.assertEqual((run/'COMPLETE.json').is_file(),active_shard==2)
                    committed_markers={p.name:p.read_bytes() for p in (run/'pairs').glob('*.json')}
                self.assertEqual([len(observed[shard]) for shard in range(3)],[2,2,2])
                observed_ids=[pair_id for batches in observed.values() for _,pair_ids in batches for pair_id in pair_ids]
                self.assertEqual(len(observed_ids),len(set(observed_ids)))
                self.assertEqual(set(observed_ids),{p['pair_id'] for p in pairs})
                for active_shard in range(3):stream.worker(args(active_shard))
            self.assertEqual([len(observed[shard]) for shard in range(3)],[2,2,2])
            self.assertTrue(all(stream.committed(run,p) for p in pairs))
            self.assertEqual({p.name:p.read_bytes() for p in (run/'pairs').glob('*.json')},committed_markers)
            self.assertEqual({p.stem for p in (run/'batches').glob('*.json')},
                             {'batch_%04d'%batch for batch in range(6)})
            self.assertFalse(any((root/'work/baseline').glob('batch_*')))
            status=json.loads((run/'status.json').read_text())
            self.assertEqual((status['completed'],status['expected']),(len(pairs),len(pairs)))
            complete=json.loads((run/'COMPLETE.json').read_text())
            self.assertEqual(complete['num_pairs'],len(pairs))

    def test_invalid_batch_shard_configuration_fails_before_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fake_worker_run(root,[41,8,92],3)
            for index,count in [(0,2),(0,0),(0,-1),(-1,3),(3,3)]:
                with self.subTest(index=index,count=count):
                    args=type('Args',(),dict(run_root=str(root),variant='baseline',
                                           batch_shard_index=index,batch_shard_count=count))()
                    with patch.object(stream,'phase') as phase:
                        with self.assertRaises((ValueError,AssertionError)):stream.worker(args)
                        phase.assert_not_called()

    def test_complete_report_has_twelve_adjusted_primary_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pairs=[dict(pair_id=i,pair_dir='pair_%04d'%i,src1='a%d'%i,src2='b%d'%i,
                        confirmation=i<6,source_component=i//2) for i in range(8)]
            stream.write(root/'plan.json',dict(pairs=pairs))
            stream.write(root/'statistical_protocol.json',dict(confirmation_count=6,bootstrap_samples=200,
                         permutation_samples=499,statistics_seed=42))
            for j,variant in enumerate(report.VARIANTS):
                for i,pair in enumerate(pairs):
                    stream.write(root/'variants'/variant/'pairs'/(pair['pair_dir']+'.json'),dict(
                        variant=variant,pair_dir=pair['pair_dir'],metrics=dict(
                            FID=100+i+j,PDV=.001+i*.00001+j*.000001,
                            PPL=3+i*.01+j*.001,EPRE=.3+i*.001+j*.0001,source_lpips=.25)))
            report.report(root,True)
            data=json.loads((root/'reports/results.json').read_text())
            primary=[r for r in data['paired_comparisons'] if r['cohort']=='confirmation_object_disjoint'
                     and r['family']=='primary_FID_PDV']
            self.assertEqual(len(primary),12)
            self.assertTrue(all(r['p_holm'] is not None and r['p_holm']>=r['p_two_sided'] for r in primary))
            self.assertTrue((root/'reports/COMPLETE.json').is_file())
            text=(root/'reports/index.html').read_text()
            self.assertIn('le 6 coppie',text)


if __name__=='__main__':unittest.main()
