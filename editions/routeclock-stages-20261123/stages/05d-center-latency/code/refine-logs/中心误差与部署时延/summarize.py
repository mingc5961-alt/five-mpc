from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.utils import get_column_letter

O=Path(__file__).resolve().parent
S=O.parent/'RC_Chronos2_差距定位'
labels={'original':'归档配管','special':'新专用管','cached':'归档配管＋单次预测内弧长缓存'}
def data_sheet(book,name,df):
    sh=book.create_sheet(name)
    sh.append(list(df.columns))
    for r in df.itertuples(index=False,name=None):sh.append([None if pd.isna(v) else v for v in r])
    return sh

def summarize():
    wb=Workbook();notes=wb.active;notes.title='试验口径'
    notes.append(['项目','说明'])
    for a in [
        ('当前硬件','Xeon E5-2686 v4 2.30 GHz，cgroup 约 8.33 核；NVIDIA TITAN V。RC 中心及误差头在 CPU；其余中心在 GPU，调度在 CPU。固定种子 7，在同一服务器相继运行。'),
        ('固定到达影子流','开发世界 0，16 用户轮询，一次更新一个用户；180 秒观测历史恢复后，固定每 62 ms 到达一个已经生成的观测，等于 5% 感知预算持续打满时的汇总到达率；并非自适应策略的实测到达分布。每组 64 次热身后 1024 次计分，FIFO、保留积压。'),
        ('计时边界','计划到达至主机形成 16 用户动作列表，含唤醒抖动、排队、模型/管、候选与全局调度；CUDA 同步到完成。传感采集、无线驱动和射频生效尚待硬件测量。62 ms 是持续到达间隔，10/20 ms 仅为参考阈值。'),
        ('配管','归档组：RC 专用尺度头、Chronos-2 原生分位数、VECTOR 和 RAFT 固定管；专用组：上轮各中心自己的双角前缀尺度 LightGBM。所有组均使用冻结权重和当前实验推理实现；缓存、启动批与模型装载不计入稳态分位数。'),
        ('准确性','合成确认集比较 40 世界各 16 用户预热末相同观测，另有 59 个实飞回放单元。门控事后角误差最优解按各期限在 IMM/路线两预测之间搜索，读取真值，仅作可表达性诊断；其位置误差另用位置最优权重。'),
        ('调整试验','在 960 条训练路线 × 3 锚点 × 4 期限上训练两个独立候选；评估 400 条留出验证路线和 40 个确认世界。新门控用当前期限特征，回归事后最优路线权重，按专家双角分歧平方加权；残差方案保留冻结 RC，修正方位、俯仰、距离，训练集各期限 95% 裁剪、输出收缩 0.5。两候选均为离线端点诊断；上线前需要全预测网格、配套管重校准及闭环重测。'),
        ('不确定性','尾时延仅是一条轨迹一次有序回放，未跨设备或重复热稳态认证。角误差改进区间按整世界/整路线重采样，同一时间窗口内的无人机或锚点始终共同采样。')
    ]:notes.append(a)
    t=[]; raw=[]
    jobs=[(m,head,O/f'{m}_{head}') for head in ('original','special') for m in ('RC','Chronos-2','VECTOR','RAFT')]
    jobs.append(('RC','cached',O/'弧长缓存/RC_original'))
    for m,head,base in jobs:
        d=json.loads(base.with_suffix('.json').read_text());r=pd.read_csv(base.with_suffix('.csv'));z=r[r.scored].copy()
        assert len(z)==1024 and max(abs(z.response_ms-z.wait_ms-z.service_ms))<1e-5
        assert (z.response_ms>62).sum()==d['late_62']
        t.append([m,labels[head],len(z),d['forecast_p50'],d['schedule_p50'],z.service_ms.mean(),d['compute_p99'],d['queue_p99'],
                  d['p50'],d['p95'],d['p99'],d['maximum'],d['late_10'],d['late_20'],d['late_62'],100*d['late_62']/len(z),
                  d['load_and_restore_ms'],d['initial_batch_with_checks_ms']])
        r.insert(0,'模型',m);r.insert(1,'误差管',labels[head]);raw.append(r)
    latency=pd.DataFrame(t,columns=['模型','管','计分事件数','预测及管 P50 ms','调度等 P50 ms','纯服务均值 ms','纯服务 P99 ms','等待 P99 ms',
                    '含排队 P50 ms','含排队 P95 ms','含排队 P99 ms','含排队最大 ms','超过 10ms 次','超过 20ms 次','超过 62ms 次','超过 62ms %',
                    '装载/恢复 ms','首次16人指令含核验 ms'])
    data_sheet(wb,'服务器实时性',latency)
    oracle=pd.read_csv(S/'跨版本归因/门控与专家上限.csv')
    data_sheet(wb,'专家和门控',oracle.groupby(['horizon_s','model'],as_index=False)[['position_m','angle_deg']].mean())
    old=pd.read_csv(S/'跨版本归因/同历史输入与训练.csv')
    data_sheet(wb,'微调对照',old.groupby(['horizon_s','model'],as_index=False)[['position_m','angle_deg']].mean())
    anchors=pd.concat([pd.read_csv(f) for f in S.glob('anchors_*.csv')])
    data_sheet(wb,'合成和实飞',anchors.groupby(['domain','model','h'],as_index=False)[['position_m','angle_deg','safe32_prefix']].mean())
    trials=pd.read_csv(O/'中心调整逐锚点.csv')
    data_sheet(wb,'调整效果',trials.groupby(['split','model','h'],as_index=False)[['position_m','angle_deg']].mean())
    by=trials[trials.split=='确认世界预热末'].groupby(['group','model','h'])
    data_sheet(wb,'确认世界轨迹分组',by.agg(端点数=('angle_deg','size'),双角误差度=('angle_deg','mean'),位置米=('position_m','mean'),
                    运动端点数=('along_abs_m','count'),切向绝对误差米=('along_abs_m','mean'),垂直于切向误差米=('cross_m','mean')).reset_index())
    effects=[]
    for split,d in trials.groupby('split'):
        for h,v in d.groupby('h'):
            mat=v.groupby(['unit','model']).angle_deg.mean().unstack()
            indices=np.random.default_rng(271828).integers(0,len(mat),(5000,len(mat)))
            for m in ['新门控','RC加残差']:
                delta=(mat[m]-mat.RC).to_numpy();lo,hi=np.quantile(delta[indices].mean(1),[.025,.975])
                effects.append([split,h,m,delta.mean(),lo,hi,int((delta<0).sum()),len(mat)])
    data_sheet(wb,'候选减RC配对差',pd.DataFrame(effects,columns=['数据','期限 s','候选','角误差差 °','95% 下限','95% 上限','改善世界或路线数','世界或路线数']))
    data_sheet(wb,'逐事件时延',pd.concat(raw,ignore_index=True))
    for s in wb:
        s.freeze_panes='C2';s.auto_filter.ref=s.dimensions
        for c in s[1]:c.fill=PatternFill('solid',fgColor='23475A');c.font=Font(color='FFFFFF',bold=True)
        for j,c in enumerate(s[1],1):
            s.column_dimensions[get_column_letter(j)].width=max(14,min(35,len(c.value)*2+1))
        if s==notes:
            s.column_dimensions['B'].width=100
            for row in list(s.rows)[1:]:row[1].alignment=Alignment(wrap_text=True,vertical='center');s.row_dimensions[row[0].row].height=50
    p=O/'中心精度与实时性实测.xlsx';wb.save(p)
    check=load_workbook(p,read_only=True);assert check['服务器实时性'].max_row==10 and check['逐事件时延'].max_row==9793;check.close()
    print(latency.round(2).to_string(index=False))
    print(pd.DataFrame(effects).round(4).to_string(index=False))
    with (O.parent/'实时实验进度.txt').open('a') as f:
        print(datetime.now().strftime('%F %T'),'中心误差与实时性完成。4 模型 × 2 配管、8192 个稳态事件，另有 1024 个 RC 弧长缓存事件，80 个预热快照的路线预测数组逐项一致；960 路线训练两个离线调整候选，400 路线验证和 40 世界配对复核完毕。配对效果、分组、真实时钟逐事件时延及目标机采时工具存于',O,file=f)
if __name__=='__main__':summarize()
