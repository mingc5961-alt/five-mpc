"""Publish failed stage logs and resume atomically validated work after interruption."""
import datetime,os,sys,time,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def alive(pid,start):
    try:
        v=Path(f'/proc/{pid}/stat').read_text().split();return v[2]!='Z' and v[21]==start
    except FileNotFoundError:return False
def main():
    pid=int(sys.argv[1]);start=Path(f'/proc/{pid}/stat').read_text().split()[21]
    while alive(pid,start):time.sleep(8)
    if (root/'report_receipt.json').is_file():return
    run=0
    while not (root/'report_receipt.json').is_file():
        run+=1
        dest=root/f'logs/recovery_{datetime.datetime.now():%Y%m%d_%H%M%S}_{run}.log'
        (root/'logs/status.txt').write_text(f'执行器退出；正在从有哈希回执的工作单元续跑。恢复日志 {dest.name}\n')
        with dest.open('wb') as log:
            proc=subprocess.run([sys.executable,'-Bu',str(root/'code/resume849.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode==0:return
        print('恢复退出',proc.returncode,dest,flush=True)
        time.sleep(min(60,5*run))
if __name__=='__main__':main()
