import hashlib
import json
from pathlib import Path
import numpy as np
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Alignment,Font,PatternFill
from openpyxl.utils import get_column_letter
from protocol import OUT,ROOT,PRIOR,MODELS,KINDS,BUDGETS,NS,SEED
from replay import reference

TITLES={'common':'公共输入尺度管','special':'专用输入尺度管','original':'归档配管'}
ORIGINAL={'RC':'RC 归档尺度管','Chronos-2':'原生分位数管','VECTOR':'固定管','RAFT':'固定管'}
CAP={'T':3.,'M':4.5,'L':9.}
NAMES=['净吞吐 Mbps','风险 %','感知 s','预算使用 %','无有效动作 %','几何超限 %','32×32 有效数据 s',
       '更新次数','切换 s','有效数据 s','无服务数据 s']
HCOLS=['期限 s','整体前缀覆盖 %','方位角半宽 °','俯仰角半宽 °']
book=Workbook()
first=book.active;first.title='读表说明'
def sheet(name,columns):
    s=book.create_sheet(name);s.append(columns);return s
def read(path):
    return json.loads(path.read_text())
def title(m,k):
    return ORIGINAL[m] if k=='original' else TITLES[k]

def record(v,b):
    air,sw,yes,no=v['ledger_s']
    assert abs(sum(v['ledger_s'])-180)<1e-6
    assert len(v['bits'])==16 and v['probes_per_user']==9000 and abs(sum(v['bits'])/180e6-v['G_mbps'])<1e-7
    assert 0<=v['unavailable_fraction']<=v['R']<=1 and abs(v['updates']*.0031-air)<1e-7 and air<=CAP[b]+.0031001
    return [v['G_mbps'],100*v['R'],air,100*air/CAP[b],100*v['unavailable_fraction'],100*(v['R']-v['unavailable_fraction']),
            v['narrow_good_data_s'],v['updates'],sw,yes,no]
def cell_path(m,k,b,w):
    return reference(m,w,b) if k=='original' else OUT/'cells'/m/k/f'{w:03}_{b}.json'

def build():
    notes=[
    ('问题','同一冻结中心和同一误差学习目标下，专用诊断能否比只读公共信息改善包络，以及能否提高净吞吐或降低闭环风险。'),
    ('规模','八个新误差头；10 个开发世界 × 三档预算 × 八种配置共 240 场新闭环；另核对四模型归档配管 120 个既有同世界回执。训练种子 7。'),
    ('公共输入','每条观测真实到达时才计数。23 个字段涵盖期限、观测位置/协方差、径向速度/标准差、过去 180 秒实测次数和最近间隔/缺口、速度、预测位移/距离、视线转角及前缀最大转角。另有两个角轴指示量。'),
    ('专用输入','RC 使用归档 60 个专家诊断字段和两轴路线/IMM 中心分歧。Chronos-2 补 12 个分位距/角宽诊断。VECTOR 补观测与预测速度、加速度、最大转向和从 256 维解码状态拟合的 8 个训练集主成分。RAFT 补三检索尺度的匹配质量、邻居离散、跨尺度方向分歧，共 33 个。'),
    ('标签与基准','所有新树学习截至 h 的逐角轴最大绝对误差的对数，祖先期限的最大值确保尺度不随期限缩小。各中心使用自己的误差与独立拟合的树。归档参照分别为 RC 归档尺度头、Chronos-2 原生分位数、VECTOR/RAFT 固定管。'),
    ('训练划分','每中心的 240 条管训练轨迹共 720 锚点，192/48 条路线内部分组选择超参数和迭代数，再在全部 240 条重训。400 条独立校准轨迹各一个锚点；400 条验证轨迹共 1200 锚点，仅用于新头冻结后的评估。'),
    ('校准与验证','按期限取双角及 20 ms 稠密时间前缀最大归一化误差，400 个校准样本取有限样本 90% 分位系数，系数随期限不减。离线实际半宽报期限终点的校准后均值，两个轴分别给出。'),
    ('同覆盖率诊断','为比较相同覆盖率下的宽度，在验证集每个期限回看 1200 个得分的第1080个顺序统计量，只用于描述性比较。全部闭环始终使用独立校准集冻结的系数。'),
    ('通信条件','16 用户共享 180 秒预热和 180 秒服务，单次观测 3.1 ms。感知上限 3 / 4.5 / 9 秒；0 dB MCS 余量，价值调度，风险尺度 1，最长期限 10 秒。'),
    ('闭环风险','每 20 ms 为每用户计一个探针，共 144000 个。风险分解为没有有效动作和已有动作但几何损失超过 3 dB。窄波束有效时间计实际数据段使用 32×32 并且损失 ≤3 dB 的时长。'),
    ('配对置信区间','同世界左减右，整世界 bootstrap 5000 次，种子 271828，相同样本索引用于所有比较。风险差单位为百分点，区间为未校正的描述性 95% 区间；每格 10 个独立世界。'),
    ('复证边界','此处为开发世界同中心的闭环比较，10 个世界和单一中心种子；离线覆盖描述原验证轨迹分布。闭环反馈会改变随后观测历史，离线覆盖与通信总风险分母不同。')]
    first.append(['条目','内容'])
    for r in notes:first.append(r)
    means=sheet('闭环均值',['模型','配置','预算','上限 s','世界数',*NAMES])
    pairs=sheet('配对效应',['左模型','左配置','右模型','右配置','预算','净吞吐差 Mbps','95% 下限','95% 上限','吞吐胜出世界数',
                               '风险差 百分点','95% 下限','95% 上限','无动作差 百分点','几何风险差 百分点','感知差 s','32×32 有效时间差 s'])
    offline=sheet('独立验证',['模型','配置','验证轨迹','锚点',*HCOLS])
    eq=sheet('同覆盖率管宽',['模型','配置','期限 s','实际覆盖 %','实际方位 °','实际俯仰 °','描述性覆盖 %','同覆盖方位 °','同覆盖俯仰 °'])
    raw=sheet('逐场闭环',['模型','配置','预算','世界','上限 s',*NAMES,'初始中心 SHA-256','来源'])
    fits=sheet('训练及校准',['模型','配置','输入列数','叶数','最小叶样本','迭代数','训练路线','校准路线','校准倍率 0.5/1/2/3/5/8/10 s','权重 SHA-256'])
    center_sheet=sheet('同历史中心误差',['模型','验证轨迹','锚点','5 s 双角误差范数均值 °','10 s 双角误差范数均值 °'])
    for m in MODELS:
        c=read(OUT/'original'/f'{m}.json')['center_5s_10s_deg']
        center_sheet.append([m,400,1200,*c])
    data={}
    for m in MODELS:
        reference_hash={}
        for k in ('original',*KINDS):
            if k=='original':
                v=read(OUT/'original'/f'{m}.json')['validation']
            else:
                fp=OUT/'heads'/m/(k+'.pkl')
                sha=hashlib.sha256(fp.read_bytes()).hexdigest()
                meta=read(OUT/'heads'/m/(k+'.json'))
                v=meta['validation']
                fits.append([m,title(m,k),meta['n_features'],*meta['chosen'],240,400,
                             ', '.join(f'{f:.5f}' for f in v['factors']),sha])
            for i,h in enumerate([.5,1,2,3,5,8,10]):
                offline.append([m,title(m,k),400,1200,h,100*v['coverage'][i],*v['angular_halfwidth_deg'][i]])
            for b in BUDGETS:
                by_world=[]
                for w in range(10):
                    path=cell_path(m,k,b,w)
                    p=read(path);r=p['result']
                    assert (p['namespace'],p['world'],p['budget'],p['model'])==(NS,w,b,m)
                    if k!='original':assert p['head']==k and p['head_sha256']==sha
                    h=r['initial_center_sha256']
                    if w in reference_hash:assert h==reference_hash[w]
                    reference_hash[w]=h
                    val=record(r,b)
                    by_world.append(val)
                    raw.append([m,title(m,k),b,w,CAP[b],*val,h,str(path)])
                a=np.array(by_world)
                data[m,k,b]=a
                means.append([m,title(m,k),b,CAP[b],10,*a.mean(0)])
    for r in read(OUT/'validation_widths.json'):
        eq.append([r['model'],title(r['model'],r['kind']),r['h'],100*r['cover'],*r['width'],100*r['equalcover'],*r['equalwidth']])
    idx=np.random.default_rng(271828).integers(0,10,(5000,10))
    contrasts=[(m,l,m,r) for m in MODELS for l,r in [('special','common'),('special','original'),('common','original')]]
    contrasts += [(m,'special','RC','special') for m in MODELS if m!='RC']
    for m,l,n,r in contrasts:
        for b in BUDGETS:
            diff=data[m,l,b]-data[n,r,b];avg=diff.mean(0)
            ci=np.quantile(diff[idx,:2].mean(1),[.025,.975],axis=0)
            pairs.append([m,title(m,l),n,title(n,r),b,avg[0],*ci[:,0],int(sum(diff[:,0]>0)),avg[1],*ci[:,1],
                          avg[4],avg[5],avg[2],avg[6]])
    for s in book:
        s.freeze_panes='C2' if s!=first else 'A2'
        s.auto_filter.ref=s.dimensions
        for c in s[1]:
            c.font=Font(color='FFFFFF',bold=True);c.fill=PatternFill('solid',fgColor='214C60')
        for col in s.columns:
            size=max([len(str(col[0].value))*1.9]+[len(str(c.value))*1.55 if isinstance(c.value,str) else 11 for c in col[1:]])
            s.column_dimensions[get_column_letter(col[0].column)].width=max(14,min(90,size+1))
            for c in col[1:]:
                if c.data_type=='n' and isinstance(c.value,(np.floating,float)):c.number_format='0.000'
        if s==first:
            for row in list(s.rows)[1:]:
                row[1].alignment=Alignment(wrap_text=True,vertical='center');s.row_dimensions[row[0].row].height=45
    dest=OUT/'八个专用与公共尺度头实验结果.xlsx'
    book.save(dest)
    check=load_workbook(dest,read_only=True)
    assert [s.max_row for s in check]==[len(notes)+1,37,46,85,17,361,9,5]
    check.close()
    return data
if __name__=='__main__':
    data=build()
    for m in MODELS:
        for k in ('original',*KINDS):
            print(m,k,[np.round(data[m,k,b].mean(0)[[0,1,2,4,5,6]],3).tolist() for b in BUDGETS])
