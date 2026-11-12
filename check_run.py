"""Verify saved preprocessing, held-out separation, metrics and checkpoint inference."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from src.data import load_clean, save_json
from src.model import MutationMLP, predict_array
from src.metrics import regression_metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default='results/run_001'); p.add_argument('--data',default='data/raw/skempi_v2.csv'); a=p.parse_args()
    torch.set_num_threads(2)
    run=Path(a.run); rows,x,audit=load_clean(a.data)
    pre=json.loads((run/'preprocessing.json').read_text())
    train=np.array([r['split']=='train' for r in rows])
    np.testing.assert_allclose(pre['mean'],x[train].mean(0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(pre['scale'],np.maximum(x[train].std(0),1),rtol=0,atol=1e-12)
    groups={s:{r['group'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    pdbs={s:{r['pdb'] for r in rows if r['split']==s} for s in groups}
    for s in groups:
        for t in groups:
            if s!=t:
                assert not groups[s]&groups[t]
                assert not pdbs[s]&pdbs[t]
    with (run/'predictions.csv').open() as f: saved=list(csv.DictReader(f))
    assert len(saved)==len(rows)
    assert [r['record_id'] for r in saved]==[r['record_id'] for r in rows]
    z=(x-np.array(pre['mean']))/np.array(pre['scale'])
    predictions=[]
    for pth in sorted(run.glob('model_seed_*.pt')):
        c=torch.load(pth,map_location='cpu',weights_only=True)
        m=MutationMLP(c['input_dim'],c['hidden'],c['dropout']); m.load_state_dict(c['state_dict'])
        predictions.append(predict_array(m,z)*pre['target_scale']+pre['target_mean'])
    pred=np.mean(predictions,axis=0)
    stored=np.array([float(r['mlp_prediction']) for r in saved])
    np.testing.assert_allclose(pred,stored,rtol=1e-6,atol=1e-6)
    test=np.array([r['split']=='test' for r in rows])
    y=np.array([r['ddg_kcal_mol'] for r in rows])
    expected=json.loads((run/'metrics.json').read_text())['test']['mlp_ensemble']
    recomputed=regression_metrics(y[test],stored[test])
    for k in ('rmse','mae','pearson','spearman','r2'):
        assert abs(recomputed[k]-expected[k])<1e-10
    save_json(run/'quality_checks.json',dict(status='passed',unit_test_count=6,
        train_only_preprocessing=True,no_pdb_overlap=True,no_metadata_group_overlap=True,
        checkpoint_reload_matches_saved_predictions=True,reported_test_metrics_match_predictions=True,
        records=len(rows),raw_sha256=audit['raw_sha256'],max_reload_prediction_difference=float(np.max(np.abs(pred-stored)))))
    print('All saved-run consistency checks passed')
if __name__=='__main__': main()
