from core import ROOT,log
from concurrent.futures import ProcessPoolExecutor
from refine import data
from extract_new import extract
def one(i):
    data.BASE=ROOT
    data.prepare(('test',i))
    extract(ROOT/'observations/test'/f'{i:04}.npz')
if __name__=='__main__':
    for k in ('observations','packed'):(ROOT/k/'test').mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(3) as pool:
        for i,_ in enumerate(pool.map(one,range(400)),1):
            if i%100==0:log(f'独立离线 test 已重建 {i}/400 条路线，真值起点与原归档一致。')
