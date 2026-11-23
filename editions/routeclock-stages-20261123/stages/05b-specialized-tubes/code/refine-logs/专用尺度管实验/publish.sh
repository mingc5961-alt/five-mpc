#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
while [[ "$(find cells -mindepth 3 -maxdepth 3 -type f -name '*.json' | wc -l)" -lt 240 ]]; do sleep 4; done
"$PY" audit_inputs.py
"$PY" report.py > logs/final_means
"$PY" make_pdf.py > logs/pdf 2>&1
/data/coding/report_tools/bin/python - <<'PY'
import pymupdf
from pathlib import Path
p=Path('专用尺度管实验分析.pdf')
with pymupdf.open(p) as doc:
    assert len(doc)==3
    for page in doc:
        text=page.get_text()
        assert len(text)>440 and 'Chronos' in text
        for block in page.get_text('blocks'):
            x0,y0,x1,y1=block[:4]
            assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1
print('PDF 已重新打开，三页内容与页面边界核验通过。')
PY
"$PY" - <<'PY'
import json
from openpyxl import load_workbook
from protocol import OUT,log
path=OUT/'八个专用与公共尺度头实验结果.xlsx'
wb=load_workbook(path,read_only=True,data_only=True)
eff=list(wb['配对效应'].values)[1:]; summary=list(wb['闭环均值'].values)[1:]
log('已全部完成。八个头均已冻结，240/240 场新闭环与 120 个同世界归档参照核验完毕。验证 5/10 秒实际双角覆盖与宽度、同经验覆盖下宽度、各档吞吐/风险/用时、窄波束有效服务与五千次世界配对重采样均已汇总。')
for model in ['RC','Chronos-2','VECTOR','RAFT']:
    new=[row for row in summary if row[0]==model and row[1]=='专用输入尺度管']
    pair=[row for row in eff if row[:4]==(model,'专用输入尺度管',model,'公共输入尺度管')]
    assert len(new)==len(pair)==3
    absolute='；'.join(f'{r[2]} {r[5]:.2f} Mbps / {r[6]:.3f}% / {r[7]:.3f} s' for r in new)
    delta=' / '.join(f'{r[5]:+.2f}' for r in pair)
    log(f'{model} 专用管（净吞吐 / 风险 / 实际感知）：{absolute}。专用减公共净吞吐（紧/中/松）{delta} Mbps。')
log(f'文件已重新打开核验，数值工作簿 {path}，三页分析报告 {OUT/"专用尺度管实验分析.pdf"}。')
PY
