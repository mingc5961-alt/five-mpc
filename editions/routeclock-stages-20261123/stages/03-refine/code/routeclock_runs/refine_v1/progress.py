"""Show the live full-campaign view or start its publisher."""
import os
import sys
from pathlib import Path

base=Path(__file__).resolve().parent
view=base.parents[1]/'refine-logs'/'实时实验进度.txt'
if __name__=='__main__':
    if len(sys.argv)==1 and view.is_file():
        print(view.read_text(),end='')
    else:
        os.execv(sys.executable,[sys.executable,str(base/'experiment_progress.py'),*sys.argv[1:]])
