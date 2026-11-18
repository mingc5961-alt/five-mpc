from pathlib import Path
import json,html,re
from report_content import B,TITLE,SUBTITLE,O
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from PIL import Image as PILImage
NAME='RouteClock_实验结果与机制综合报告'
(O/'analysis/report_blocks.json').write_text(json.dumps({'title':TITLE,'subtitle':SUBTITLE,'blocks':B},ensure_ascii=False,indent=2))
# HTML with navigation and printable content
H=[f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{TITLE}</title><style>
@font-face{{font-family:Noto;src:url('assets/NotoSansSC-Regular.ttf')}}
body{{font-family:Noto,'Microsoft YaHei',sans-serif;color:#253044;margin:0;background:#eef2f6;line-height:1.85}}main{{max-width:1080px;margin:auto;padding:44px 48px;background:white}}h1{{font-size:32px;line-height:1.4}}h2{{color:#006d77;margin-top:48px;border-bottom:2px solid #d8e7e8;padding-bottom:10px}}h3{{margin-top:30px}}p{{text-align:justify}}.sub,.caption{{color:#586778;font-size:13px}}table{{border-collapse:collapse;width:100%;font-size:12px;line-height:1.65;margin:12px 0 26px}}th,td{{border:1px solid #dbe2e9;padding:8px;vertical-align:top;overflow-wrap:anywhere}}th{{background:#e9f2f3;text-align:left}}tr:nth-child(even){{background:#f7f9fb}}figure{{margin:22px 0}}img{{max-width:100%;height:auto}}nav{{background:#f1f7f7;padding:18px}}nav a{{display:block;color:#006d77}}.tablewrap{{overflow:auto}}@media print{{body{{background:white}}main{{padding:0}}nav{{display:none}}h2{{break-before:page}}tr,figure{{break-inside:avoid}}}}
</style><main><h1>{TITLE}</h1><p class="sub">{SUBTITLE}</p><p>本报告包含完整正面和负面结果；所有图表可在 figures 与 tables 目录复用。参数建议均未执行。</p><nav>''']
for i,b in enumerate(B):
 if b['type']=='heading':H.append(f'<a href="#s{i}">{html.escape(b["text"])}</a>')
H.append('</nav>')
for i,b in enumerate(B):
 typ=b['type']
 if typ in ['heading','subheading']:tag='h2' if typ=='heading' else 'h3';H.append(f'<{tag} id="s{i}">{html.escape(b["text"])}</{tag}>')
 elif typ=='paragraph':H.append('<p>'+html.escape(b['text'])+'</p>')
 elif typ=='figure':H.append(f'<figure><a href="{b["path"]}"><img src="{b["path"]}" alt="{html.escape(b["caption"])}"></a><figcaption class="caption">{html.escape(b["caption"])}</figcaption></figure>')
 elif typ=='table':
  title=html.escape(b['title']);title=f'<a href="tables/{title}">{title}</a>' if title.endswith('.csv') else title
  H.append(f'<p class="caption">{title}</p><div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in b['headers'])+'</tr></thead><tbody>')
  for row in b['rows']:H.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>')
  H.append('</tbody></table></div>')
H.append('</main></html>');(O/(NAME+'.html')).write_text('\n'.join(H),encoding='utf-8')
# DOCX editable
D=Document();sec=D.sections[0];sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65);sec.left_margin=Inches(.65);sec.right_margin=Inches(.65)
for sn in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 s=D.styles[sn];s.font.name='Noto Sans SC';s._element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Sans SC')
D.styles['Normal'].font.size=Pt(10);D.styles['Normal'].paragraph_format.line_spacing=1.3
D.add_heading(TITLE,0);D.add_paragraph(SUBTITLE,'Subtitle');D.add_paragraph('完整结果与机制分析｜可编辑稿。建议汇报时按第十六节取舍。')
D.add_heading('目录',1)
for b in B:
 if b['type']=='heading':D.add_paragraph(b['text'])
D.add_page_break()
for b in B:
 typ=b['type']
 if typ=='heading':D.add_heading(b['text'],1)
 elif typ=='subheading':D.add_heading(b['text'],2)
 elif typ=='paragraph':D.add_paragraph(b['text'])
 elif typ=='figure':D.add_picture(str(O/b['path']),width=Inches(6.9));D.add_paragraph(b['caption'],'Caption')
 elif typ=='table':
  D.add_paragraph(b['title'],'Caption');tb=D.add_table(rows=1,cols=len(b['headers']));tb.style='Light Shading Accent 1'
  trpr=tb.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
  for cell,value in zip(tb.rows[0].cells,b['headers']):cell.text=value
  for row in b['rows']:
   for cell,value in zip(tb.add_row().cells,row):cell.text=value
  for row in tb.rows:
   for cell in row.cells:
    for par in cell.paragraphs:
     par.paragraph_format.space_after=Pt(3)
     for run in par.runs:run.font.size=Pt(8)
  D.add_paragraph()
footer=sec.footer.paragraphs[0];footer.alignment=2;footer.add_run('RouteClock 实验报告  |  ');field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
D.save(O/(NAME+'.docx'))
# PDF with Chinese font embedded; headers repeat on split tables.
pdfmetrics.registerFont(TTFont('NotoSC',str(O/'assets/NotoSansSC-Regular.ttf')))
styles=getSampleStyleSheet();base=dict(fontName='NotoSC',wordWrap='CJK',textColor=colors.HexColor('#253044'))
st={
 'p':ParagraphStyle('zhp',fontSize=9.5,leading=16,spaceAfter=9,**base),
 'h':ParagraphStyle('zhh',fontSize=16,leading=24,spaceBefore=20,spaceAfter=12,keepWithNext=True,**base),
 's':ParagraphStyle('zhs',fontSize=12,leading=19,spaceBefore=14,spaceAfter=8,keepWithNext=True,**base),
 't':ParagraphStyle('zht',fontSize=7.3,leading=11,**base),
 'c':ParagraphStyle('zhc',fontSize=8,leading=12,spaceAfter=9,**base),
 'title':ParagraphStyle('zhtitle',fontSize=23,leading=34,spaceAfter=18,**base)}
def para(v,key='p'):return Paragraph(html.escape(str(v)),st[key])
flow=[Spacer(1,35),para(TITLE,'title'),para(SUBTITLE),para('完整结果与机制分析。正面结论、负面结论、诊断证据和待验证假设分别列明。'),Spacer(1,12),para('目录','h')]
for b in B:
 if b['type']=='heading':flow.append(para(b['text'],'c'))
flow.append(PageBreak());W=A4[0]-84
for b in B:
 typ=b['type']
 if typ=='heading':flow.append(para(b['text'],'h'))
 elif typ=='subheading':flow.append(para(b['text'],'s'))
 elif typ=='paragraph':flow.append(para(b['text']))
 elif typ=='figure':
  path=O/b['path'];im=PILImage.open(path);w,h=im.size;scale=min(W/w,290/h);flow.append(Image(str(path),width=w*scale,height=h*scale));flow.append(para(b['caption'],'c'))
 elif typ=='table':
  flow.append(para(b['title'],'c'));data=[[para(v,'t') for v in b['headers']]]+[[para(v,'t') for v in row] for row in b['rows']];n=len(b['headers']);weights=[1]*n
  if n==2:weights=[.7,1.7]
  elif n==3:weights=[.8,1.5,1.4]
  elif n==4 and 'CI' not in str(b['headers']):weights=[.8,1.2,1,1.2]
  tb=Table(data,colWidths=[W*x/sum(weights) for x in weights],repeatRows=1,hAlign='LEFT');tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3eef0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')]),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#ccd6de')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));flow.extend([tb,Spacer(1,12)])
def page(canvas,doc):
 canvas.setFont('NotoSC',8);canvas.setFillColor(colors.HexColor('#687787'));canvas.drawString(42,24,'RouteClock | 最终结果 2026-11-18');canvas.drawRightString(A4[0]-42,24,str(doc.page))
pdf=SimpleDocTemplate(str(O/(NAME+'.pdf')),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=40,title=TITLE,author='RouteClock experiment analysis');pdf.build(flow,onFirstPage=page,onLaterPages=page)
print('Reports generated:',len(B),'blocks,',sum(len(b.get('text','')) for b in B),'Chinese/text characters')
