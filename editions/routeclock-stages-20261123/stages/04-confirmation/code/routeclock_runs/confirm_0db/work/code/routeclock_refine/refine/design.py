"""Enumerate the planned 2,280 unique synthetic closed-loop cells."""
import csv
from .data import BASE
MAIN=('F','I','P','RC','chronos','CR-G','vector','moirai')
LEARNED=('chronos','moirai','vector')
def cells():
    for w in range(20):
        for c in ('T','L'):
            for m in MAIN:yield 'validation',w,c,m,'default',7,5
        for m in ('F','I','RC'):yield 'validation',w,'M',m,'default',7,5
        for margin in (3,7):
            for m in ('F','RC'):yield 'validation',w,'T',m,'default',7,margin
    for w in range(70):
        for m in MAIN:yield 'confirmation',w,'T',m,'default',7,5
        for m in (*LEARNED,'RC'):yield 'confirmation',w,'T',m,'fixed',7,5
        yield 'confirmation',w,'T','CR_raw','default',7,5
        for c in ('M','L'):
            for m in ('F','I','RC'):yield 'confirmation',w,c,m,'default',7,5
        for seed in (11,19):
            for tube in ('default','fixed'):yield 'confirmation',w,'T','vector',tube,seed,5
        for m in LEARNED:yield 'confirmation',w,'T',m,'lightgbm',7,5

if __name__=='__main__':
    rows=list(cells())
    assert len(set(rows))==len(rows)==2280
    with (BASE/'planned_cells.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['phase','world','condition','model','tube','seed','mcs_margin_db']);w.writerows(rows)
