import argparse
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.data import load_clean, feature_names, save_json, write_csv
from src.model import MutationMLP, predict_array
from src.metrics import regression_metrics, cluster_bootstrap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/skempi_v2.csv')
    parser.add_argument('--out', default='results/run_001')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seeds', default='11,22,33')
    a = parser.parse_args()
    out = Path(a.out)
    if out.exists(): raise SystemExit('Output directory exists. Use a new --out directory.')
    out.mkdir(parents=True)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows, x, audit = load_clean(a.data)
    y = np.array([r['ddg_kcal_mol'] for r in rows], dtype=float)
    split = np.array([r['split'] for r in rows])
    masks = {s:split == s for s in ('train','validation','test')}
    group_sets = {s:{r['group'] for r in rows if r['split']==s} for s in masks}
    for s in masks:
        for t in masks:
            if s != t: assert not group_sets[s] & group_sets[t]
    mean = x[masks['train']].mean(0)
    scale = np.maximum(x[masks['train']].std(0), 1.0)
    z = (x-mean)/scale
    ym, ys = float(y[masks['train']].mean()), float(y[masks['train']].std())
    assert ys > 0
    normalized_y = (y-ym)/ys
    save_json(out/'data_audit.json', audit)
    write_csv(out/'split_manifest.csv', rows)
    save_json(out/'preprocessing.json', dict(feature_names=feature_names(), mean=mean.tolist(),
                                            scale=scale.tolist(), target_mean=ym, target_scale=ys))
    train_ix, val_ix = masks['train'], masks['validation']
    # Baseline 1: no features; baseline 2: same inputs, ridge regularization chosen on validation only.
    baseline_mean = np.repeat(ym, len(y))
    xx = z[train_ix]
    xtx, xty = xx.T@xx, xx.T@(y[train_ix]-ym)
    best_ridge = None
    ridge_trials = []
    for alpha in (.1,1.,10.,100.,1000.):
        coef = np.linalg.solve(xtx + alpha*np.eye(x.shape[1]), xty)
        pred = z@coef+ym
        score = regression_metrics(y[val_ix],pred[val_ix])['rmse']
        ridge_trials.append(dict(alpha=alpha,validation_rmse=score))
        if best_ridge is None or score < best_ridge[0]: best_ridge=(score,alpha,coef,pred)
    ridge_pred = best_ridge[3]
    np.savez(out/'ridge_baseline.npz', coef=best_ridge[2], intercept=ym)
    save_json(out/'ridge_validation.json',dict(trials=ridge_trials,selected_alpha=best_ridge[1]))
    all_predictions, histories, runs = [], [], []
    for seed in [int(s) for s in a.seeds.split(',')]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = MutationMLP(x.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        criterion = nn.MSELoss()
        dataset = TensorDataset(torch.tensor(z[train_ix],dtype=torch.float32), torch.tensor(normalized_y[train_ix],dtype=torch.float32))
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset,batch_size=128,shuffle=True,generator=generator,num_workers=0)
        best, stale, best_state, best_epoch = float('inf'), 0, None, 0
        for epoch in range(1,a.epochs+1):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(bx),by)
                assert torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            ptrain = predict_array(model,z[train_ix])*ys+ym
            pval = predict_array(model,z[val_ix])*ys+ym
            tr = regression_metrics(y[train_ix],ptrain)['rmse']
            vr = regression_metrics(y[val_ix],pval)['rmse']
            histories.append(dict(seed=seed,epoch=epoch,train_rmse=tr,validation_rmse=vr))
            if vr < best-1e-5:
                best=vr; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); stale=0
            else: stale += 1
            if stale >= a.patience: break
        assert best_state is not None
        model.load_state_dict(best_state)
        torch.save(dict(state_dict=best_state,input_dim=x.shape[1],hidden=64,dropout=.1,seed=seed),out/f'model_seed_{seed}.pt')
        all_predictions.append(predict_array(model,z)*ys+ym)
        runs.append(dict(seed=seed,best_epoch=best_epoch,epochs_run=epoch,validation_rmse=best))
        print(f'seed={seed} best_epoch={best_epoch} validation_RMSE={best:.4f}',flush=True)
    ensemble = np.mean(all_predictions,axis=0)
    predictions = []
    for i,r in enumerate(rows):
        item = dict(r, mean_prediction=float(baseline_mean[i]),ridge_prediction=float(ridge_pred[i]),
                    mlp_prediction=float(ensemble[i]))
        for j,run in enumerate(runs): item[f"seed_{run['seed']}_prediction"]=float(all_predictions[j][i])
        predictions.append(item)
    write_csv(out/'predictions.csv',predictions)
    write_csv(out/'learning_curves.csv',histories)
    metric_results={}
    for split_name,mask in masks.items():
        metric_results[split_name]={}
        for name,pred in [('mean',baseline_mean),('ridge',ridge_pred),('mlp_ensemble',ensemble)]:
            result=regression_metrics(y[mask],pred[mask])
            if split_name=='test':
                result['cluster_bootstrap']=cluster_bootstrap(y[mask],pred[mask],np.array([r['group'] for r in rows])[mask])
            metric_results[split_name][name]=result
    abmask=masks['test'] & np.array([r['is_antibody'] for r in rows])
    metric_results['antibody_test_diagnostic']={name:regression_metrics(y[abmask],pred[abmask]) for name,pred in [('mean',baseline_mean),('ridge',ridge_pred),('mlp_ensemble',ensemble)]}
    per_seed=[]
    for run,pred in zip(runs,all_predictions):
        per_seed.append(dict(run,test=regression_metrics(y[masks['test']],pred[masks['test']])))
    save_json(out/'metrics.json',metric_results)
    save_json(out/'training_runs.json',per_seed)
    group_metrics=[]
    for group in sorted(group_sets['test']):
        mask=np.array([r['group']==group for r in rows])
        group_metrics.append(dict(group=group,**regression_metrics(y[mask],ensemble[mask])))
    write_csv(out/'test_group_metrics.csv',group_metrics)
    save_json(out/'run_config.json',dict(epochs=a.epochs,patience=a.patience,threads=a.threads,
              model='451 -> 64 -> 32 -> 1 MLP, ReLU and dropout 0.1',optimizer='AdamW',lr=.001,
              weight_decay=.001,batch_size=128,seeds=[r['seed'] for r in runs],
              selection='validation early stopping; primary ensemble of all three seeds fixed before testing',
              trainable_parameters=sum(p.numel() for p in model.parameters()),torch_version=torch.__version__,
              numpy_version=np.__version__,device='cpu',elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(audit=audit,test=metric_results['test'],elapsed_seconds=time.perf_counter()-started),indent=2),flush=True)

if __name__=='__main__': main()
