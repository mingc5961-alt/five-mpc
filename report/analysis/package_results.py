"""Package immutable results + report + source/weights, with SHA-256 and ZIP CRC verification."""
from pathlib import Path
import zipfile,hashlib,json,time,os
ROOT=Path('/data/chen');REPORT=ROOT/'teacher_report_20261118';RUN=ROOT/'formal_execution/run_20261117_001732';SCI=ROOT/'routeclock_server'
OUT=ROOT/'RouteClock_导师汇报与完整实验结果_20261118.zip';TMP=OUT.with_suffix('.zip.partial');TOP='RouteClock_20261118'
files=[];excluded=[]
def addtree(root,prefix,exclude=()):
 for p in sorted(root.rglob('*')):
  rel=p.relative_to(root)
  if any(x in set(exclude) for x in rel.parts):
   if p.is_file():excluded.append(str(p))
   continue
  if p.is_file():files.append((p,f'{TOP}/{prefix}/{rel.as_posix()}'))
addtree(REPORT,'report',('__pycache__',))
addtree(RUN,'formal_execution/run_20261117_001732',('__pycache__',))
# Keep all scientific source/config/dependencies/model weights; omit historical run outputs.
addtree(SCI,'code/routeclock_server',('__pycache__','.git','runs','results','.pytest_cache'))
for p in sorted((ROOT/'formal_execution').glob('*.py')):files.append((p,f'{TOP}/code/formal_execution/{p.name}'))
for dirname in ['server_planning_20261117']:
 addtree(ROOT/dirname,f'provenance/{dirname}',('__pycache__',))
readme='''RouteClock 导师汇报与完整实验结果（2026-11-18）

先看 report/RouteClock_实验结果与机制综合报告.pdf；需要修改用同名 DOCX；HTML 可检索并打开 CSV 与图。
本包包含本轮全部正式实验输出、2,400 条离线任务数据、训练 bundle、保留的完整/紧凑日志、765 场网络策略运行结果、143 次刷新回放、完成与范围审计、复核分析脚本、科学代码及预训练权重。
765=645 FIVE +120 MPC，不代表765个独立物理世界。45个TabPFN作业按批准范围延后，未伪记为完成。

目录：
 report/：报告、40张派生表、14张PNG及SVG图、分析和渲染脚本。
 formal_execution/run_20261117_001732/：完整正式运行目录，未人为剔除负结果。
 code/routeclock_server/：科学代码、配置、依赖及权重；历史runs/results目录未纳入，避免把旧demo混为正式结果。
 code/formal_execution/：本轮作业编排、并发控制、审计与测试代码。
 provenance/：服务器实验规划与并发调整记录。
 FILE_MANIFEST.tsv：逐文件字节数与SHA-256；不包含清单自身。
 PACKAGE_METADATA.json：打包范围与校验口径。

复核：按照 report/analysis/report_requirements.txt 准备报告环境。先运行 analyze_results.py <本包正式运行目录> <本包report目录>，再运行 analyze_additional.py <本包正式运行目录>，最后 generate_diagram.py 和 build_report.py。分析不会修改原始数据。
迁移：本包包含代码和模型，但不包含Python虚拟环境、CUDA驱动、旧迁移压缩包。环境需按正式运行目录requirements-resolved.txt和科学代码requirements重建，核对绝对路径与模型缓存。原代码包含不同历史子项目，实际报告使用five与server_full两条线。不要直接重启旧supervisor PID或把旧锁文件当作活进程。
未发现科学源文件变更：报告核查154个记录的源码/配置哈希全部一致。ZIP CRC检查及完整文件哈希用于传输完整性，不代表科学结论自动正确。
压缩包外同名.sha256.txt用于校验整个ZIP。解压后可按FILE_MANIFEST.tsv逐文件重算SHA-256。
'''
start=time.time();total=sum(p.stat().st_size for p,_ in files);print(json.dumps({'stage':'start','files':len(files),'raw_GiB':total/2**30},ensure_ascii=False),flush=True)
manifest=[];done=0;last=0
with zipfile.ZipFile(TMP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
 z.writestr(TOP+'/开始阅读.txt',readme)
 manifest.append((len(readme.encode()),hashlib.sha256(readme.encode()).hexdigest(),TOP+'/开始阅读.txt'))
 for i,(p,arc) in enumerate(files,1):
  before=p.stat();digest=hashlib.sha256();count=0
  with p.open('rb') as src,z.open(arc,'w',force_zip64=True) as dst:
   while buf:=src.read(4*1024*1024):dst.write(buf);digest.update(buf);count+=len(buf)
  after=p.stat();assert before.st_size==after.st_size==count and before.st_mtime_ns==after.st_mtime_ns, f'File changed while packing: {p}'
  manifest.append((count,digest.hexdigest(),arc));done+=count
  if time.time()-last>20:
   print(json.dumps({'stage':'compress','file':i,'files':len(files),'percent_bytes':round(100*done/total,1),'elapsed_s':round(time.time()-start),'current':str(p.relative_to(ROOT))},ensure_ascii=False),flush=True);last=time.time()
 meta={'generated':'2026-11-18','source_run':str(RUN),'payload_files':len(manifest),'raw_bytes':sum(x[0] for x in manifest),'compression':'ZIP64 deflate level3','science_scope':'645 FIVE + 120 MPC, 45 deferred','excluded_historical_or_cache_files':len(excluded),'symlinks':'File symlinks materialized as content for portability; duplicate linked weights may be stored twice.','integrity':'SHA-256 for every payload file; ZIP testzip after completion; separate full archive SHA-256.'}
 z.writestr(TOP+'/PACKAGE_METADATA.json',json.dumps(meta,ensure_ascii=False,indent=2));z.writestr(TOP+'/FILE_MANIFEST.tsv','bytes\tsha256\tpath\n'+'\n'.join(f'{n}\t{d}\t{a}' for n,d,a in manifest)+'\n')
print(json.dumps({'stage':'verify_crc','archive_GiB':TMP.stat().st_size/2**30,'elapsed_s':round(time.time()-start)}),flush=True)
with zipfile.ZipFile(TMP) as z:bad=z.testzip();assert bad is None,bad;assert len(z.infolist())==len(manifest)+2
os.replace(TMP,OUT);sha=hashlib.sha256()
with OUT.open('rb') as src:
 while b:=src.read(8*1024*1024):sha.update(b)
shahex=sha.hexdigest();OUT.with_suffix('.zip.sha256.txt').write_text(f'{shahex}  {OUT.name}\n')
result={'archive':str(OUT),'bytes':OUT.stat().st_size,'GiB':OUT.stat().st_size/2**30,'payload_files':len(manifest),'sha256':shahex,'crc_verified':True,'source_files_stable':True,'elapsed_s':round(time.time()-start)}
(ROOT/'RouteClock_20261118_打包校验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
