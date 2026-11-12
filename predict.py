import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.data import features
from src.model import MutationMLP,predict_array

def main():
    p=argparse.ArgumentParser(description='Predict mutation delta-delta-G, not absolute affinity or de novo peptide binding.')
    p.add_argument('--run',default='results/run_001')
    p.add_argument('--mutations',required=True,help='Cleaned SKEMPI syntax, e.g. AA10V or AA10V,GB20D')
    p.add_argument('--locations',default='COR',help='COR,RIM,SUP,SUR,INT,UNKNOWN per mutation')
    p.add_argument('--partner1',default='A'); p.add_argument('--partner2',default='B')
    p.add_argument('--temperature',type=float,default=298.)
    a=p.parse_args(); run=Path(a.run)
    pre=json.loads((run/'preprocessing.json').read_text())
    x=features(a.mutations,a.locations,a.partner1,a.partner2,a.temperature)[None,:]
    z=(x-np.array(pre['mean']))/np.array(pre['scale'])
    values=[]
    for model_file in sorted(run.glob('model_seed_*.pt')):
        ckpt=torch.load(model_file,map_location='cpu',weights_only=True)
        m=MutationMLP(ckpt['input_dim'],ckpt['hidden'],ckpt['dropout']); m.load_state_dict(ckpt['state_dict'])
        values.append(float(predict_array(m,z)[0])*pre['target_scale']+pre['target_mean'])
    if not values: raise SystemExit('No trained model files found')
    print(json.dumps(dict(predicted_ddg_kcal_mol=float(np.mean(values)),seed_predictions=values,
                         convention='negative favors tighter binding; positive favors weaker binding',
                         scope='mutation-only exploratory model; no antigen coordinates or sequence embeddings; not experimental validation'),indent=2))
if __name__=='__main__': main()
