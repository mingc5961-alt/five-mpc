import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue_compare import OUT,append

def run(i):
    with (OUT/'cells'/f'bridge_{i}.log').open('a') as f:
        subprocess.run([sys.executable,str(OUT/'compare.py'),'bridge',str(i)],stdout=f,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':
    append('另排 20 场对照。把 11 月 19 日原版 C 专用 LightGBM 头和校准乘子移植到统一平台；在旧世界与 RC 配对，并比较本轮按名义 SNR 或保守 SNR 选规划档位。两种规则的空口传输评分完全相同。')
    with ThreadPoolExecutor(2) as ex:
        fs={ex.submit(run,i):i for i in range(20)}
        failed=[]
        for k,f in enumerate(as_completed(fs),1):
            try:f.result()
            except Exception:failed.append(fs[f])
            if k%5==0:append(f'历史误差头及规划选档补充　已结束 {k}/20，需复查 {len(failed)}。')
    print('failed',failed,flush=True);sys.exit(bool(failed))
