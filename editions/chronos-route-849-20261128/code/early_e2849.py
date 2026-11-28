from design849 import *
from parallel849 import busy
def early_endpoint(job):
    tag=busy('开发闭环提前运行')
    try:return execute_endpoint(job)
    finally:tag.unlink(missing_ok=True)
def main():
    from gate849 import poolmap
    for a in ('time','route'):
        p=OUT/f'E2/labels/VALUE-{a}.json'
        while not p.is_file():time.sleep(2)
        assert get(p)['sha256']==sha_file(p.with_suffix('.txt'))
    supplement=len(sys.argv)>1 and sys.argv[1]=='supplement'
    ids=(1405,1404) if supplement else (1407,1406)
    scales=(.75,) if supplement else (.75,.6)
    jobs=[('development','development',k,r,s,m) for k in ids for r,s in MATRIX for m in scales]
    poolmap(early_endpoint,jobs,14 if supplement else 30)
if __name__=='__main__':main()
