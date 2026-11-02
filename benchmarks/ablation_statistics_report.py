#!/usr/bin/env python3
"""Paired uncertainty and multiplicity-aware comparisons of four fixed models."""
import argparse
import csv
from datetime import datetime, timezone
import html
import itertools
import json
import math
from pathlib import Path

import numpy as np

VARIANTS=('baseline','gate_only','prior_only','full')
METRICS=('FID','PDV','PPL','EPRE')


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1.,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def bootstrap_means(values,samples,seed,clusters=None):
    """Paired means; connected-component bootstrap retains equal pair weight."""
    values=np.asarray(values,dtype=float)
    if clusters is None:clusters=np.arange(len(values))
    _,ids=np.unique(clusters,return_inverse=True)
    n=int(ids.max())+1
    sums=np.zeros((n,)+values.shape[1:]);np.add.at(sums,ids,values)
    counts=np.bincount(ids)
    rng=np.random.default_rng(seed)
    output=[]
    for start in range(0,samples,128):
        chosen=rng.integers(n,size=(min(128,samples-start),n))
        total=sums[chosen].sum(axis=1)
        denom=counts[chosen].sum(axis=1)
        output.append(total/denom.reshape((-1,)+(1,)*(values.ndim-1)))
    return np.concatenate(output)


def paired_permutation_p(differences,samples,seed):
    """Two-sided paired sign swaps, shared Monte Carlo draws for all contrasts.

    Tests exchangeability of the two method labels within each pair. The +1
    adjustment includes the observed allocation and prevents zero MC p-values.
    """
    differences=np.asarray(differences,dtype=float)
    observed=np.abs(differences.mean(axis=0))
    exceed=np.zeros(differences.shape[1],dtype=np.int64)
    rng=np.random.default_rng(seed)
    tolerance=np.maximum(1e-15,np.abs(observed)*1e-12)
    for start in range(0,samples,2048):
        signs=rng.integers(0,2,size=(min(2048,samples-start),len(differences)),dtype=np.int8)*2-1
        perm=np.abs(signs @ differences/len(differences))
        exceed+=(perm>=observed-tolerance).sum(axis=0)
    return (exceed+1)/(samples+1)


def statistics(values,pairs,protocol,cohort,clusters=None,tests=False):
    boot=bootstrap_means(values,protocol['bootstrap_samples'],protocol['statistics_seed'],clusters)
    summaries=[]
    for i,name in enumerate(VARIANTS):
        for j,metric in enumerate(METRICS):
            x=values[:,i,j];ci=np.quantile(boot[:,i,j],[.025,.975])
            summaries.append(dict(cohort=cohort,variant=name,metric=metric,num_pairs=len(x),
                                  mean=float(x.mean()),std=float(x.std()),median=float(np.median(x)),
                                  ci95_low=float(ci[0]),ci95_high=float(ci[1])))
    comparisons=[];differences=[]
    for i,k in itertools.combinations(range(4),2):
        # Difference is first minus second; negative favors first.
        for j,metric in enumerate(METRICS):
            d=values[:,i,j]-values[:,k,j];b=boot[:,i,j]-boot[:,k,j]
            ci=np.quantile(b,[.025,.975]);sd=float(d.std(ddof=1))
            change=100*(values[:,i,j].mean()/values[:,k,j].mean()-1)
            comparisons.append(dict(cohort=cohort,first=VARIANTS[i],second=VARIANTS[k],metric=metric,
                family='primary_FID_PDV' if metric in ('FID','PDV') else 'secondary_exploratory_PPL_EPRE',
                num_pairs=len(d),mean_difference=float(d.mean()),percent_change=float(change),
                ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                paired_effect_dz=float(d.mean()/sd) if sd>0 else None,
                first_lower=int((d<0).sum()),second_lower=int((d>0).sum()),ties=int((d==0).sum()),
                p_two_sided=None,p_holm=None,reject_holm_005=None))
            differences.append(d)
    if tests:
        p=paired_permutation_p(np.stack(differences,axis=1),protocol['permutation_samples'],protocol['statistics_seed'])
        for family in ['primary_FID_PDV','secondary_exploratory_PPL_EPRE']:
            indices=[i for i,r in enumerate(comparisons) if r['family']==family]
            corrected=holm(p[indices])
            for i,padj in zip(indices,corrected):
                comparisons[i].update(p_two_sided=float(p[i]),p_holm=float(padj),reject_holm_005=bool(padj<.05))
    return summaries,comparisons


def report(root,require_complete=False):
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text());protocol=json.loads((root/'statistical_protocol.json').read_text())
    data={}
    for variant in VARIANTS:
        data[variant]={}
        for path in (root/'variants'/variant/'pairs').glob('*.json'):
            row=json.loads(path.read_text());assert row['variant']==variant
            data[variant][row['pair_dir']]=row
    common=set.intersection(*(set(data[v]) for v in VARIANTS))
    expected={p['pair_dir'] for p in plan['pairs']}
    assert common<=expected
    complete=common==expected and all(len(data[v])==len(expected) for v in VARIANTS)
    if require_complete and not complete:raise ValueError('Four-way complete pairs {}/{}'.format(len(common),len(expected)))
    out=root/'reports';out.mkdir(exist_ok=True)
    progress=dict(complete=complete,paired_complete=len(common),expected=len(expected),variants={v:len(data[v]) for v in VARIANTS},updated_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'progress.json',progress)
    if not common:
        print(json.dumps(progress));return
    pairs=[p for p in plan['pairs'] if p['pair_dir'] in common]
    values=np.array([[[data[v][p['pair_dir']]['metrics'][m] for m in METRICS] for v in VARIANTS] for p in pairs])
    assert np.isfinite(values).all()
    for p in pairs:
        D=[data[v][p['pair_dir']]['metrics']['source_lpips'] for v in VARIANTS]
        assert max(D)-min(D)<1e-5,('Canonical sources differ',p['pair_dir'],D)
    flat=[dict(variant=v,pair_id=p['pair_id'],pair_dir=p['pair_dir'],src1=p['src1'],src2=p['src2'],
               confirmation=p['confirmation'],source_component=p['source_component'],
               **data[v][p['pair_dir']]['metrics']) for p in pairs for v in VARIANTS]
    write_csv(out/'per_pair_metrics.csv',flat)
    if not complete:
        write_csv(out/'interim_means_descriptive_only.csv',[
            dict(variant=v,num_pairs=len(pairs),**{m:float(values[:,i,j].mean()) for j,m in enumerate(METRICS)})
            for i,v in enumerate(VARIANTS)])
        print(json.dumps(progress));return
    summaries,comparisons=[],[]
    confirm=np.array([p['confirmation'] for p in pairs])
    assert int(confirm.sum())==protocol['confirmation_count']
    selected=[p for p in pairs if p['confirmation']]
    assets=[s for p in selected for s in (p['src1'],p['src2'])]
    assert len(assets)==len(set(assets))
    for cohort,mask,clusters,tests in [
        ('confirmation_object_disjoint',confirm,None,True),
        ('all_500_component_bootstrap',np.ones(len(pairs),dtype=bool),np.array([p['source_component'] for p in pairs]),False)]:
        ss,cc=statistics(values[mask],pairs,protocol,cohort,clusters,tests);summaries+=ss;comparisons+=cc
    write_csv(out/'summary.csv',summaries);write_csv(out/'paired_comparisons.csv',comparisons)
    write_json(out/'results.json',dict(progress=progress,protocol=protocol,summary=summaries,paired_comparisons=comparisons))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    colors=['#666666','#56B4E9','#D55E00','#0072B2']
    for ax,metric in zip(axes,['FID','PDV']):
        rows=[r for r in summaries if r['cohort']=='confirmation_object_disjoint' and r['metric']==metric]
        scale=1000 if metric=='PDV' else 1
        y=np.array([r['mean']*scale for r in rows]);lo=np.array([r['ci95_low']*scale for r in rows]);hi=np.array([r['ci95_high']*scale for r in rows])
        for i in range(4):
            ax.errorbar(i,y[i],yerr=[[y[i]-lo[i]],[hi[i]-y[i]]],fmt='o',color=colors[i],capsize=4)
        ax.set_xticks(range(4),VARIANTS,rotation=20);ax.set_ylabel('PDV x 1000' if metric=='PDV' else metric)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('{} object-disjoint confirmation pairs; 95% paired bootstrap intervals'.format(int(confirm.sum())))
    fig.savefig(out/'primary_metrics.png',dpi=180);plt.close(fig)
    def table(rows,columns):
        return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(format(r[c],'.6g') if isinstance(r[c],float) else str(r[c]))+'</td>' for c in columns)+'</tr>' for r in rows)+'</table>'
    page='''<!doctype html><html lang="it"><meta charset="utf-8"><title>Ablation paired evaluation</title>
<style>body{font:15px system-ui;margin:30px;color:#17263c}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}th{background:#edf3f7}img{max-width:1000px;width:100%}</style>
<h1>Baseline, gate, prior, full: valutazione appaiata</h1>
<p>Tutti i __PAIRS__ test pair; CFG SS 5 / SLat 1.5; 7 stati; LPIPS 64 viste; FID concat 1000 viste/stato.</p>
<p>L'inferenza primaria usa le __CONFIRM__ coppie di conferma con oggetti tutti distinti e assenti dal pilot, selezionate prima dei nuovi risultati.
I test appaiati sono a due code; Holm controlla i 12 confronti primari FID/PDV. Gli intervalli al 95% sono marginali, non simultanei.
PPL/EPRE hanno una famiglia esplorativa separata. Le statistiche su tutte le 500 coppie ricampionano componenti con sorgenti condivise e restano descrittive.
I risultati sono condizionati ai checkpoint; i budget dei training non sono uniformi e non sono stati ripetuti gli addestramenti.</p>
<p><a href="summary.csv">Medie e intervalli</a> · <a href="paired_comparisons.csv">Differenze e test corretti</a> · <a href="per_pair_metrics.csv">Dati per coppia</a></p>
<img src="primary_metrics.png"><h2>Statistiche</h2>'''
    page+=table(summaries,['cohort','variant','metric','num_pairs','mean','std','ci95_low','ci95_high'])
    page+='<h2>Confronti primari: negativo favorisce il primo modello</h2>'
    page+=table([r for r in comparisons if r['cohort']=='confirmation_object_disjoint' and r['family']=='primary_FID_PDV'],
                ['first','second','metric','mean_difference','percent_change','ci95_low','ci95_high','p_holm','first_lower','second_lower'])
    page=page.replace('__PAIRS__',str(len(pairs))).replace('__CONFIRM__',str(int(confirm.sum())))
    (out/'index.html').write_text(page+'</html>')
    write_json(out/'COMPLETE.json',dict(num_pairs=len(pairs),confirmation_pairs=int(confirm.sum()),variants=list(VARIANTS),completed_utc=datetime.now(timezone.utc).isoformat()))
    print('COMPLETE:',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',required=True);p.add_argument('--require-complete',action='store_true')
    args=p.parse_args();report(args.run_root,args.require_complete)
