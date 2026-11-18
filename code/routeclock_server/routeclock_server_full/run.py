"""中文命令行：prepare → train → simulate → report；all 自动串行执行并可续跑。"""
import os
for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[var]="1"
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from mpc.data import ROOT
from mpc.setup_run import prepare, validate, log, save_json

@contextmanager
def writer(out):
    """一个输出目录只允许一个写进程。操作系统在异常退出后也会释放文件锁。"""
    with (out/".writer.lock").open("a+b") as f:
        f.seek(0,2)
        if f.tell()==0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            raise SystemExit("这个输出目录已有运行进程。查看进度请使用 status。") from e
        yield

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["all","prepare","train","simulate","report","verify","status"])
    parser.add_argument("--config",type=Path,help="默认读取已有运行清单；新目录使用 demo.json。")
    parser.add_argument("--out",type=Path,default=ROOT/"runs/demo")
    parser.add_argument("--workers",type=int,default=2,help="独立数据场景的并行进程数")
    parser.add_argument("--rerun",action="store_true",help="simulate 阶段即使已有账本也重新仿真")
    parser.add_argument("--models",type=Path,help="simulate 使用此前独立训练目录中的模型，新测试目录可为空")
    args=parser.parse_args()
    if args.rerun and args.stage!="simulate":
        parser.error("--rerun 仅搭配 simulate")
    if args.models and args.stage!="simulate":
        parser.error("--models 仅用于 simulate")
    out=args.out.resolve()
    if args.stage=="status":
        print((out/"status.json").read_text(encoding="utf8") if (out/"status.json").exists() else "尚无进度记录")
        return
    if args.workers<1:
        parser.error("--workers 必须至少为 1")
    out.mkdir(exist_ok=True,parents=True)
    with writer(out):
        execute(args,out,parser)

def execute(args,out,parser):
    # 运行配置与输出绑定，避免测试场景被另一套参数覆盖。
    manifest=out/"config.json"
    path=args.config or (manifest if manifest.exists() else ROOT/"configs/demo.json")
    c=json.loads(path.read_text(encoding="utf-8-sig"))
    validate(c)
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf8"))!=c:
        parser.error("此输出目录已有不同配置，请通过 --out 指定新的运行目录。")
    save_json(manifest,c)
    try:
        if args.stage in ("all","prepare"):
            prepare(c,out,args.workers)
        if args.stage in ("all","train","simulate","report"):
            from mpc.experiment import train, simulate, report
            for name,func in [("train",train),("simulate",simulate),("report",report)]:
                if args.stage in ("all",name):
                    if name=="simulate":
                        prepare(c,out,args.workers,("test",))
                        func(c,out,args.rerun,args.models)
                    else:
                        if name=="train" and not (out/"models.json").exists():
                            prepare(c,out,args.workers,("gate","train","tube_cal","cal"))
                        func(c,out)
        if args.stage in ("all","verify"):
            from mpc.audit import verify
            verify(c,out)
        log(out,"完成",args.stage)
    except Exception as error:
        log(out,"失败",repr(error))
        raise

if __name__=="__main__":
    main()
