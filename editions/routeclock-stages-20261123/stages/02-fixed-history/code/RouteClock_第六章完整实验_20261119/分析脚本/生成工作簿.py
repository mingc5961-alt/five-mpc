from pathlib import Path
import pandas as pd
from openpyxl.utils import get_column_letter

root = Path(__file__).resolve().parents[1]
tables = root/'数据表'
primary = root/'主实验归档/runs/paired600'
sources = {name:tables/(name+'.csv') for name in ['测试工况全指标','测试方法配对','吞吐差对称分解','验证机制配对','验证工况全指标','逐场指标1500']}
sources.update({'等间隔配对':primary/'matched_spacing.csv','调度配对':primary/'scheduler_paired.csv'})
new_names=['固定历史工况全指标','固定历史方法配对','固定历史与原协议配对','高成本两档交互','历史及服务效应','固定历史等间隔对照','同动作F预算对照','固定历史调度对照','起点预测密集减原历史','合并逐场指标2080']
sources={**{name:tables/(name+'.csv') for name in new_names},**sources}
with pd.ExcelWriter(root/'实验数据与配对结果.xlsx',engine='openpyxl') as writer:
    for title,path in sources.items():
        df=pd.read_csv(path)
        df.to_excel(writer,sheet_name=title,index=False,freeze_panes=(1,2))
        ws=writer.sheets[title]
        ws.auto_filter.ref=ws.dimensions
        for idx,name in enumerate(df.columns,1):
            cells=df[name].dropna().head(100)
            width=len(name)+3
            if not pd.api.types.is_numeric_dtype(cells):
                width=max(width,*(len(str(v))+2 for v in cells),12)
            ws.column_dimensions[get_column_letter(idx)].width=min(66,max(14,width))
            if pd.api.types.is_float_dtype(df[name]):
                for cell in ws[get_column_letter(idx)][1:]:
                    cell.number_format='#,##0.0000'
