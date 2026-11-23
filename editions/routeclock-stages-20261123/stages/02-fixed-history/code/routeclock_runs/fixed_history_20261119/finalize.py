"""Build, verify and atomically publish the revision, retaining the complete original."""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import subprocess
import time
import zipfile

here=Path(__file__).resolve().parent
chen=Path('/data/chen')
stage=chen/'.routeclock_report_fixed_history_staging'
live=chen/'RouteClock_第六章完整实验_20261119'
backup=chen/'RouteClock_第六章完整实验_20261119_初版'
science='/data/chen/server_assessment/venv/bin/python'
renderer='/data/coding/report_tools/bin/python'
state=here/'report_status.json'
def mark(phase,**extra):
    temp=state.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(phase=phase,at=time.time(),**extra),ensure_ascii=False,indent=2))
    temp.replace(state)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def manifest(root,files,dest):
    text=''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in sorted(files))
    dest.write_text(text)
    for line in text.splitlines():
        value,name=line.split('  ',1)
        assert sha(root/name)==value
def run():
    mark('等待全部闭环与起点诊断')
    while json.loads((here/'status.json').read_text())['stage']!='simulation_complete' or len(list((here/'boundary').glob('*.json')))!=30:
        time.sleep(15)
    mark('逐场审计、统一数据、重算置信区间')
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    for py,script in [(science,'固定历史分析.py'),(science,'生成工作簿.py'),(renderer,'生成报告.py')]:
        subprocess.run([py,str(stage/'分析脚本'/script)],cwd=stage,env=env,check=True)
    mark('校验报告和工作簿')
    subprocess.run([renderer,'-c',"""import pymupdf,sys,zipfile
from pathlib import Path
p=Path(sys.argv[1]);pdf=next(p.glob('RouteClock*分析报告.pdf'))
doc=pymupdf.open(pdf)
assert len(doc)>=14 and all(len(page.get_text())>70 for page in doc)
assert '控制起始历史后' in ''.join(page.get_text() for page in doc)
with zipfile.ZipFile(pdf.with_suffix('.docx')) as z:
 assert z.testzip() is None and '固定起始历史后的修正' in z.read('word/document.xml').decode()
""",str(stage)],check=True)
    subprocess.run([science,'-c',"""import sys
from pathlib import Path
import pandas as pd
from openpyxl import load_workbook
p=Path(sys.argv[1]);w=load_workbook(p/'实验数据与配对结果.xlsx',read_only=True)
assert w['合并逐场指标2080'].max_row==2081
for name,size in [('合并逐场指标2080',2080),('逐机资源合并',33280),('固定历史逐场指纹',581),('服务起点预测逐机',25920)]:
 assert len(pd.read_csv(p/'数据表'/(name+'.csv')))==size
""",str(stage)],check=True)
    jobs=json.loads((here/'manifest.json').read_text())['new']
    fixedfiles=[p for p in here.iterdir() if p.is_file() and p.suffix in {'.py','.json'} and p not in {state}]
    for folder in ['policies','boundary','logs']:
        fixedfiles.extend(p for p in (here/folder).rglob('*') if p.is_file() and p.name!='finalize.log')
    for job in jobs:
        fixedfiles.extend(p for p in Path(job['out']).iterdir() if p.is_file())
    manifest(here,fixedfiles,here/'SHA256SUMS')
    files=[p for p in stage.rglob('*') if p.is_file() and not p.is_symlink() and p.name!='报告交付_SHA256SUMS' and '__pycache__' not in p.parts]
    manifest(stage,files,stage/'报告交付_SHA256SUMS')
    for link in stage.iterdir():
        if link.is_symlink():
            target=link.resolve(strict=True)
            link.unlink();link.symlink_to(os.path.relpath(target,stage),target_is_directory=target.is_dir())
    # Atomic directory exchange on the data volume. The old tree stays on disk throughout.
    if backup.exists():raise ValueError('Previous version path occupied')
    swap=ctypes.CDLL(None,use_errno=True).renameat2
    if swap(-100,os.fsencode(stage),-100,os.fsencode(live),2):
        err=ctypes.get_errno();raise OSError(err,os.strerror(err))
    stage.rename(backup)
    assert (live/'固定历史补充实验').resolve()==here and (backup/'报告交付_SHA256SUMS').exists()
    mark('报告与实验已合并并通过校验',report=str(live),previous=str(backup),new_runs=len(jobs))
if __name__=='__main__':
    try:run()
    except Exception as exc:
        mark('需要检查',error=repr(exc));raise
