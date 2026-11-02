#!/usr/bin/env python3
"""Bounded-disk evaluation of one fixed model on the frozen full-test plan."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


STOP = False
CHILD = None


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.tmp.'+uuid.uuid4().hex)
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def copy_atomic(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name('.'+destination.name+'.tmp.'+uuid.uuid4().hex)
    with Path(source).open('rb') as src,temporary.open('wb') as dst:
        shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
    temporary.replace(destination)


def scores(result, plan):
    assert result['num_generated_frames']==5 and result['num_views']==64
    assert result['lpips_backbone']=='vgg' and result['render_backend']=='gaussian'
    assert result.get('full_path_includes_reference_endpoints') is True
    protocol=result['fid_protocol']
    assert protocol['views_per_state']==1000
    # Selected frame identities are saved for every FID aggregation mode.
    # selected_states contains optional per-state FID scores and is absent
    # when the evaluator computes only the concatenated-state score.
    indices=protocol.get('selected_indices')
    if (not isinstance(indices,list) or len(indices)!=2
            or any(type(i) is not int or not 0<=i<5 for i in indices)
            or len(set(indices))!=2):
        raise ValueError('FID needs two distinct selected intermediate indices in [0, 4]')
    selected_alphas=protocol.get('selected_alphas')
    if not isinstance(selected_alphas,list) or len(selected_alphas)!=2:
        raise ValueError('FID selected alpha values must match the two selected indices')
    if len(plan['alphas'])!=5 or len(result['alphas'])!=5:
        raise ValueError('The frozen Interp protocol requires five intermediate alpha values')
    if any(not math.isclose(float(a),plan['alphas'][i],rel_tol=0.,abs_tol=1e-9)
           for i,a in zip(indices,selected_alphas)):
        raise ValueError('FID selected alpha values disagree with the frozen plan')
    if protocol.get('reference_samples')!=2000 or protocol.get('generated_samples')!=2000:
        raise ValueError('Concatenated FID requires 2000 reference and 2000 generated views')
    if 'concatenated' not in protocol.get('aggregation_modes',[]):
        raise ValueError('The frozen statistical evaluation requires concatenated FID')
    assert all(abs(a-b)<1e-9 for a,b in zip(result['alphas'],plan['alphas']))
    agg=result['aggregate_across_views']
    values={public:float(agg[key]['mean']) for public,key in [
        ('FID','fid_endpoint_manifold'),('PDV','pdv_full'),
        ('PPL','ppl_full_normalized_reference_endpoints'),('source_lpips','reference_endpoint_lpips')]}
    D=values['source_lpips']
    if not D>0:raise ValueError('Source LPIPS must be positive for EPRE')
    curve=result['weighted_endpoint_curve']
    assert len(curve)==5
    points=[]
    for row,alpha in zip(curve,plan['alphas']):
        assert abs(float(row['alpha'])-alpha)<1e-9
        t=1-alpha;d1=float(row['lpips_src1_mean']);d2=float(row['lpips_src2_mean'])
        e=.5*((d1/D-t)**2+(d2/D-(1-t))**2)
        points.append(dict(progress=t,alpha=alpha,lpips_src1=d1,lpips_src2=d2,profile_squared_error=e))
    xs=[0.]+[r['progress'] for r in points]+[1.]
    ys=[0.]+[r['profile_squared_error'] for r in points]+[0.]
    values['EPRE']=math.sqrt(sum((b-a)*(u+v)/2 for a,b,u,v in zip(xs,xs[1:],ys,ys[1:])))
    assert all(math.isfinite(v) for v in values.values())
    return values,points


def committed(run, pair):
    path=run/'pairs'/(pair['pair_dir']+'.json')
    if not path.is_file():return False
    value=json.loads(path.read_text())
    assert (value['pair_id'],value['src1'],value['src2'],value['pair_seed'])==(pair['pair_id'],pair['src1'],pair['src2'],pair['pair_seed'])
    for field in ['FID','PDV','PPL','EPRE']:
        assert math.isfinite(value['metrics'][field])
    for item in value['retained_files']:
        target=run/item['path']
        if not target.is_file() or target.stat().st_size!=item['bytes'] or digest(target)!=item['sha256']:
            raise RuntimeError('Committed output failed integrity check: '+str(target))
    return True


def commit_pair(root, variant, pair, cell, plan):
    from PIL import Image
    run=root/'variants'/variant
    source=cell/'metrics/morphflow'/pair['pair_dir']
    result=json.loads((source/'metrics.json').read_text())
    values,points=scores(result,plan)
    assert result['sequence_name']==pair['pair_dir']
    retained=[]
    # Preserve complete numerical diagnostics, without large Inception arrays.
    files=[p for p in source.iterdir() if p.is_file() and p.suffix in ('.json','.csv')]
    assert any(p.name=='full_per_view.csv' for p in files)
    for path in files:
        target=run/'cfg_ss_5_slat_1.5/metrics/morphflow'/pair['pair_dir']/path.name
        copy_atomic(path,target)
        retained.append(dict(path=str(target.relative_to(run)),bytes=target.stat().st_size,sha256=digest(target)))
    keys=['src1']+[('a%.6f'%a).replace('.','p') for a in plan['alphas']]+['src2']
    for key in keys:
        path=source/'images'/key/'025.png'
        with Image.open(path) as image:
            assert image.format=='PNG' and image.size==(512,512);image.verify()
        target=run/'cfg_ss_5_slat_1.5/metrics/morphflow'/pair['pair_dir']/'images'/key/'025.png';copy_atomic(path,target)
        retained.append(dict(path=str(target.relative_to(run)),bytes=target.stat().st_size,sha256=digest(target)))
    row=dict(variant=variant,pair_id=pair['pair_id'],pair_dir=pair['pair_dir'],src1=pair['src1'],src2=pair['src2'],
             pair_seed=pair['pair_seed'],confirmation=pair['confirmation'],pilot=pair['pilot'],
             source_component=pair['source_component'],metrics=values,endpoint_curve=points,
             evaluation_signature=result['evaluation_signature'],fid_protocol=result['fid_protocol'],
             committed_utc=now(),retained_files=retained)
    write(run/'pairs'/(pair['pair_dir']+'.json'),row)
    assert committed(run,pair)
    return row


def release_work(root, variant, cell, pairs):
    run=root/'variants'/variant
    assert all(committed(run,pair) for pair in pairs)
    expected=(root/'work'/variant).resolve()
    resolved=cell.resolve()
    if not resolved.is_relative_to(expected) or resolved==expected or cell.is_symlink():
        raise ValueError('Refusing to remove anything outside the new work-batch directory')
    shutil.rmtree(cell)


def stop(signum,frame):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass


def phase(root,variant,label,environment,repository,arguments):
    global CHILD
    if STOP:raise InterruptedError('Walltime/signal: resume remaining pairs in next allocation')
    project=root/'source_snapshot/MorphFlow'
    env=os.environ.copy()
    folder=root/'logs'/('job_'+env.get('SLURM_JOB_ID','local'))/variant
    env.update(TRELLIS_REPO=str(repository),PYTHONPATH=str(repository)+':'+str(project),
               PATH=str(environment/'bin')+':'+env.get('PATH',''),
               MORPHFLOW_MESH_REPAIR_SCRIPT=str(project/'benchmarks/threeway_textured_benchmark.py'),
               MORPHFLOW_MESH_REPAIR_LOG_DIR=str(folder/'mesh_repair'),
               MORPHFLOW_RENDER_REPAIR_CONFIG=str(root/'runtime/render_repair.json'))
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'mesh_repair').mkdir(exist_ok=True)
    print('[{}] {} START {}'.format(now(),variant,label),flush=True)
    with (folder/(label+'.out')).open('a') as stdout,(folder/(label+'.err')).open('a') as stderr:
        CHILD=subprocess.Popen([str(environment/'bin/python')]+[str(a) for a in arguments],
                               cwd=project,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        while CHILD.poll() is None:
            time.sleep(1)
            if STOP:
                try:CHILD.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
                raise InterruptedError('Interrupted; durable pair commits are retained')
        code=CHILD.returncode;CHILD=None
    if code:
        print((folder/(label+'.err')).read_text()[-6000:],file=sys.stderr,flush=True)
        raise RuntimeError('{} failed with status {}; {}'.format(label,code,folder))
    print('[{}] {} DONE {}'.format(now(),variant,label),flush=True)


def batch_shard(args,manifest):
    index=getattr(args,'batch_shard_index',0)
    count=getattr(args,'batch_shard_count',1)
    if count<1 or not 0<=index<count:
        raise ValueError('Batch shard index must be within the positive shard count')
    if count!=manifest.get('batch_shards',1):
        raise ValueError('Batch shard count disagrees with the frozen run manifest')
    return index,count


def update_status(run,variant,done,pairs,assigned,shard_index,shard_count):
    job=os.environ.get('SLURM_JOB_ID')
    if shard_count>1:
        write(run/('status_shard_%03d_of_%03d.json'%(shard_index,shard_count)),
              dict(variant=variant,completed=len(done),expected=len(assigned),
                   shard_index=shard_index,shard_count=shard_count,updated_utc=now(),job_id=job))
    # Pair records are published atomically after their retained files.
    # Serialize writers so the shared progress count never goes backwards.
    with (run/'.status.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        names={p['pair_dir'] for p in pairs}
        completed=sum(p.stem in names for p in (run/'pairs').glob('*.json'))
        write(run/'status.json',dict(variant=variant,completed=completed,expected=len(pairs),
              shard_count=shard_count,updated_utc=now(),job_id=job))


def _worker(args):
    root=Path(args.run_root).resolve();variant=args.variant
    plan=json.loads((root/'plan.json').read_text())
    manifest=json.loads((root/'run_manifest.json').read_text())
    shard_index,shard_count=batch_shard(args,manifest)
    assert variant in manifest['variants'] and manifest['cfg_ss']==5. and manifest['cfg_slat']==1.5
    run=root/'variants'/variant;run.mkdir(parents=True,exist_ok=True)
    for flow,pin in manifest['checkpoints'][variant].items():
        stat=Path(pin['path']).stat()
        assert (stat.st_size,stat.st_mtime_ns)==(pin['bytes'],pin['mtime_ns']),flow
    original=Path(os.environ['ORIGINAL_ENV']);metric=Path(os.environ['METRIC_ENV'])
    trellis=root/'source_snapshot/TRELLIS';render=root/'source_snapshot/MorphAny3D'
    phase(root,variant,'runtime_validation',original,trellis,['-c',
        'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.__version__,torch.cuda.get_device_name(0))'])
    phase(root,variant,'mesh_smoke',original,trellis,[root/'runtime/mesh_channel_blocks.py'])
    phase(root,variant,'gaussian_smoke',metric,render,['-c',
        'import runpy,sys;runpy.run_path(sys.argv[1])["smoke_gpu"]()',root/'source_snapshot/MorphFlow/slurm/prepare_gaussian_rasterizer.py'])
    pairs=plan['pairs']
    batch_size=manifest['batch_size']
    starts=[i for i in range(0,len(pairs),batch_size) if (i//batch_size)%shard_count==shard_index]
    assigned=[p for i in starts for p in pairs[i:i+batch_size]]
    done={p['pair_dir'] for p in assigned if committed(run,p)}
    status=lambda:update_status(run,variant,done,pairs,assigned,shard_index,shard_count)
    status()
    for index in starts:
        if STOP:raise InterruptedError('Resume on next allocation')
        scheduled=pairs[index:index+batch_size]
        remaining=[p for p in scheduled if p['pair_dir'] not in done]
        cell=root/'work'/variant/('batch_%04d'%(index//batch_size))
        if not remaining:
            if cell.exists():release_work(root,variant,cell,scheduled)
            continue
        if shutil.disk_usage(root).free < 12*shard_count*1024**3:
            raise RuntimeError('Insufficient free space for parallel temporary batches')
        cell.mkdir(parents=True,exist_ok=True)
        # A process can stop between durable commits within a batch. Remove
        # only the already committed pairs from this temporary batch before
        # replacing its plan, so the evaluator cannot discover stale sequences.
        for pair in scheduled:
            if pair['pair_dir'] in done:
                assert committed(run,pair)
                for folder in [cell/'morphflow'/pair['pair_dir'],
                               cell/'metrics/morphflow'/pair['pair_dir'],
                               cell/'references_raw'/pair['pair_dir']]:
                    if folder.exists():shutil.rmtree(folder)
        reduced=dict(plan,pairs=remaining,num_pairs=len(remaining),requested_num_pairs=len(remaining))
        write(cell/'plan.json',reduced)
        tag='batch_%04d'%(index//batch_size)
        phase(root,variant,tag+'_generation',original,trellis,[root/'runtime/resume_threeway_mesh.py','morphflow',
              '--run-root',cell,'--data-root',os.environ['DATA_ROOT_HOST'],
              '--metadata',root/'input_snapshot/metadata_test.json','--source-images-root',os.environ['SOURCE_IMAGES_ROOT_HOST'],
              '--source-image-filename','','--checkpoint-path',run/'checkpoints/ss/checkpoint.pt',
              '--slat-checkpoint-path',run/'checkpoints/slat/checkpoint.pt','--steps','25','--slat-steps','25',
              '--cfg-scale','5','--slat-cfg-scale','1.5','--mixed-precision','bf16','--trellis-model','image_large'])
        phase(root,variant,tag+'_metrics',metric,render,[root/'runtime/resume_threeway_render_cache.py','shard',
              '--run-dir',cell/'morphflow','--output-dir',cell/'metrics/morphflow','--stage','all',
              '--shard-index','0','--num-shards','1','--num-views','64','--resolution','512',
              '--camera-distribution','hammersley','--render-backend','gaussian','--appearance','materials',
              '--lpips-net','vgg','--lpips-batch-size','4','--fid-num-views','1000','--fid-batch-size','8',
              '--fid-selection-seed','42','--fid-aggregation','concatenated',
              '--reference-cache-root',cell/'reference_metrics','--device','cuda'])
        for pair in remaining:
            commit_pair(root,variant,pair,cell,plan);done.add(pair['pair_dir']);status()
            print('[{}] {} shard {}/{} COMMITTED {}/{} {}'.format(
                  now(),variant,shard_index+1,shard_count,len(done),len(assigned),pair['pair_dir']),flush=True)
        write(run/'batches'/(tag+'.json'),dict(pairs=[p['pair_id'] for p in scheduled],committed_utc=now()))
        release_work(root,variant,cell,scheduled)
    assert len(done)==len(assigned)
    if shard_count>1:
        write(run/('COMPLETE_shard_%03d_of_%03d.json'%(shard_index,shard_count)),
              dict(variant=variant,num_pairs=len(done),shard_index=shard_index,shard_count=shard_count,completed_utc=now()))
    if all(committed(run,p) for p in pairs):
        write(run/'COMPLETE.json',dict(variant=variant,num_pairs=len(pairs),completed_utc=now()))


def worker(args):
    root=Path(args.run_root).resolve()
    shard_index,shard_count=batch_shard(args,json.loads((root/'run_manifest.json').read_text()))
    run=root/'variants'/args.variant
    run.mkdir(parents=True,exist_ok=True)
    name='.worker.lock' if shard_count==1 else '.worker.shard_%03d_of_%03d.lock'%(shard_index,shard_count)
    with (run/name).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _worker(args)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',required=True)
    p.add_argument('--variant',required=True,choices=['baseline','gate_only','prior_only','full'])
    p.add_argument('--batch-shard-index',type=int,default=0)
    p.add_argument('--batch-shard-count',type=int,default=1)
    args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:worker(args)
    except InterruptedError as error:
        print(str(error),file=sys.stderr,flush=True);sys.exit(75)
