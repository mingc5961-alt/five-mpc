"""Small pre-training check of full-prefix replay against the snapshot scorer."""
from labels849 import *
def test():
    w,wm=world('development',1400);cache=Cache(w,wm,'development');s=Engine(w,cache,.75,True,'EDF-native')
    for i in range(16):
        s.preservice(i);s.buy(i,i,heads=False)
    s.preservice(16);r=cf(s,16,1,True)
    put(OUT/'audits/early_branch.json',r);print(r['total_delta'],r['slow_max_bin_delta'])
if __name__=='__main__':test()
