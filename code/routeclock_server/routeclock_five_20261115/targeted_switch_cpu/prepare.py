"""Fresh physical namespaces. Use the verified observation and feature pipeline unchanged."""
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'gate_diagnosis'))
import extract
OUT=HERE/'results'
extract.OUT=OUT
def jobs():
    # Group identity is mission % 6 in the simulator. Preserve configured proportions.
    for split,counts in [('target_pool',(16,16,16,16,16,16)),('target_val',(6,4,4,4,3,3)),
                         ('target_test',(12,8,8,8,6,6))]:
        for k in range(max(counts)):
            for g,limit in enumerate(counts):
                if k<limit:yield split,k*6+g
if __name__=='__main__':
    (OUT/'routes').mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=5) as pool:
        for j,name in enumerate(pool.map(extract.extract,jobs())):
            if j%12==11:print(j+1,name,flush=True)

