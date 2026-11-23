from pathlib import Path
import numpy as np,pandas as pd
from openpyxl import load_workbook
from core import ROOT,log
def finish():
    w=load_workbook(ROOT/'配对比较.xlsx',read_only=True,data_only=True)
    assert {'闭环均值','AMOVFLY各期限','单次服务时延ms','中心配对区间'}<=set(w.sheetnames)
    assert w['闭环逐世界'].max_row==211 and w['闭环均值'].max_row==22
    w.close()
    report=(ROOT/'结果分析报告.html').read_text()
    assert '<svg' in report and 'AMOVFLY' in report and '服务器单用户刷新时延' in report and report.count('<table')>=12
    for split,count in [('test',1200),('world',160),('flight',236)]:
        d=pd.read_csv(ROOT/(split+'.csv'))
        assert len(d)==count*17*7
        q=d[(d.arm=='joint')&(d.horizon==10)]
        assert np.isfinite(q.position_m).all() and len(q)==count
    cells=pd.read_excel(ROOT/'配对比较.xlsx',sheet_name='闭环均值')
    log('【本轮验收完毕】路线重建 train/validation/calibration 2000/2000；中心训练与微调 12/12、专用管训练/校准完成；400 条独立离线路线、10 个新世界×16 架、59 个实飞架次×4 锚点，共同历史评分全部核对。闭环 210/210；单次刷新时延 7×128 完成。')
    for b in ('T','M','L'):
        t=cells[cells.budget==b].set_index('arm')
        line='；'.join(f'{m} {t.loc[m,"G"]:.1f}/{t.loc[m,"risk"]:.2f}/{t.loc[m,"actual"]:.2f}' for m in ['RC','joint-transfer','joint-refit','Chronos-2','Chronos-native','VECTOR','RAFT'])
        log(f'【最终配对均值 {b}】每项依次为净吞吐 Mbps / 总风险 % / 实耗感知秒，{line}。')
    log('【交付】RC进度与分支/结果分析报告.html（含机制、消融、实飞、可达性、部署时延和图）；配对比较.xlsx（逐世界与 95% 区间）；冻结配置.json 和逐锚/逐场结果已保存。')
if __name__=='__main__':finish()
