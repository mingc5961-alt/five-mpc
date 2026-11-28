"""Exercise deferred rows on a training world with a pilot checkpoint."""
from concurrent.futures import ProcessPoolExecutor
import json,multiprocessing
import torch
from history import put
from paired_grid import BASE,load_checked
from physical_inputs import sha_file
from policy_env import initialize
from slots import compute_slot
OUT=BASE/'training_path_checks'
def one(row):
    import learned_matrix as lm
    from aoi_policy import Policy
    from unittest.mock import patch
    selected=json.loads((BASE/'learning/hyperparameters.json').read_bytes())
    protocol=lm.ROWS[row][0]
    path=BASE/'learning/trials'/selected[protocol]['run']/'model_4000.pt'
    net=Policy().eval();net.load_state_dict(torch.load(path,weights_only=True))
    w,meta=load_checked('train',1003);q=json.loads((BASE/'summary.json').read_bytes())['K']
    with compute_slot(),patch.object(lm,'OUT',OUT),patch.object(lm,'policy',return_value=(net,sha_file(path))):
        lm.series(w,meta['physical_sha256'],'train',1003,row,53,3,q,[.45,.75],OUT)
    return row
def main():
    initialize()
    from confirmation import plan
    from paired_design import capacity
    from geometry_panel import one as panel
    for s in ('receipts','active','streams'):(OUT/s).mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(6,mp_context=multiprocessing.get_context('fork')) as p:
        fs=[p.submit(one,r) for r in ['P02','Q03','Q04','Q07','Q08']];g=p.submit(panel,('train',1000))
        for f in fs:f.result()
        panel_check=g.result()
    rows=[json.loads(p.read_bytes()) for p in (OUT/'receipts').glob('*.json')]
    q=[r for r in rows if r['row'].startswith('Q')]
    assert len(rows)==10 and len(q)==8 and len({r['event_sha256'] for r in q})==1
    examples={f"{s['scenario']}/{s['row']}":{'weights':{'.60':.6,'1.0':.4}} for s in capacity() if s['world_first']=='1400'}
    jobs,_=plan(dict(core_worlds=24,development_policies=examples))
    assert sum(map(len,jobs.values()))==2376
    put(OUT/'summary.json',dict(physics_split='training only',model='selected pilot',training_matrix_receipts=10,
        identical_joint_event_streams=True,full_capacity_example=2376,geometry_pipeline=panel_check,formal_endpoints=0))
if __name__=='__main__':main()
