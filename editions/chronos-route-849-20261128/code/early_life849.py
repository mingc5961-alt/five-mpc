"""Use completed training/validation labels while development counterfactual diagnostics finish."""
from labels849 import trainvalue
from design849 import *
from gate849 import poolmap
def main():
    assert len(list((OUT/'E2/labels/train').glob('*/*/state.json')))==256 and len(list((OUT/'E2/labels/train_validation').glob('*/*/state.json')))==16
    for side in ('time','route'):
        p=OUT/f'E2/labels/VALUE-{side}.json'
        if p.exists():assert sha_file(p.with_suffix('.txt'))==get(p)['sha256']
        else:trainvalue(side)
    poolmap(confirm_job,[('panel',('development','development',k)) for k in range(1400,1408)],8)
if __name__=='__main__':main()
