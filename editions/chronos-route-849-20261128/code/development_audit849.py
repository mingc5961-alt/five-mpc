"""Audit complete development worlds as soon as every prespecified scale has finished."""
from audit849 import *
def main():
    remaining=set(range(1400,1408))
    while remaining:
        ready=[k for k in sorted(remaining) if len(list((OUT/'E2/receipts/development').glob(f'{k}_*.json')))==40]
        if not ready:time.sleep(8);continue
        for k in ready:
            audit_grid_world(('development',k));remaining.remove(k);print('audited',k,flush=True)
if __name__=='__main__':main()
