"""Run exact declared jobs, explicitly selected; model errors fail the job."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routeclock.config import load_config,dump_json
from routeclock.experiments import experiment_matrix,single_action_matrix,run_job
from routeclock.foundation import ChronosAdapter
p=argparse.ArgumentParser();p.add_argument('--bundles',required=True);p.add_argument('--out',required=True);p.add_argument('--block',required=True);p.add_argument('--device',default='cuda');p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=910);a=p.parse_args()
jobs=[j for j in experiment_matrix()+single_action_matrix() if j['block']==a.block and a.start<=j['job_id']<a.stop];adapter=None
for job in jobs:
    if job['method'] in ['C','CR'] and adapter is None:adapter=ChronosAdapter(a.device)
    print('RUN',job,flush=True);result=run_job(load_config(),job,a.bundles,a.out,adapter);print(result,flush=True)
