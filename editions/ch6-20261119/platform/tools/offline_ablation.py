import sys,argparse,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routeclock.dataset import load_anchors
from routeclock.training import load_bundle,fit_oof_gates
from routeclock.ablations import action_diagnostics,fixed_center_ablation,residual_gate_ablation
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
bundle=load_bundle(a.bundle);splits=[load_anchors(Path(a.data)/f'{s}.pkl') for s in ['train','validation','calibration','test']]
train,gates=fit_oof_gates(*splits,bundle['seed']);splits[0]=train
for center in ['RC','CR']:
    if center not in gates:continue
    for mode in ['shuffle','no_path']:
        fixed_center_ablation(splits,center,bundle['kind'],a.device,bundle['seed'],mode,Path(a.out)/f'{center}_{mode}.json')
action_diagnostics(splits[-1],bundle,bundle['config'],Path(a.out)/'actions_A3.json')
raw=[load_anchors(Path(a.data)/f'{s}.pkl') for s in ['train','validation','calibration','test']]
residual_gate_ablation(raw,bundle['seed'],Path(a.out)/'residual_gate.json')
