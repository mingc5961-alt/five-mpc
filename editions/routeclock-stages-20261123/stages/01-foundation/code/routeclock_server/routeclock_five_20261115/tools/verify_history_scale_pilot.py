"""Check matched futures, nested observations, finite predictions and exported metrics."""
import csv,json,pickle
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]/'runs/history_scale_pilot'
with (root/'records.pkl').open('rb') as f:records=pickle.load(f)
with (root/'metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(records)==60 and len(rows)==300
for task in {a['diagnostics']['task'] for a in records}:
    for origin in [240,300]:
        cases=sorted([a for a in records if a['diagnostics']['task']==task and a['diagnostics']['origin_s']==origin],key=lambda a:a['diagnostics']['history_s'])
        assert [a['diagnostics']['history_s'] for a in cases]==[12,30,60,120,240]
        for a in cases:
            np.testing.assert_array_equal(a['truth'],cases[-1]['truth'])
            mask=cases[-1]['history_times']>=a['history_times'][0]
            np.testing.assert_array_equal(a['observations'],cases[-1]['observations'][mask])
            assert a['history_times'][-1]==origin
            for method,pred in a['predictions'].items():
                assert np.isfinite(pred).all()
                row=next(r for r in rows if r['task']==task and float(r['origin_s'])==origin and int(r['history_s'])==a['diagnostics']['history_s'] and r['method']==method)
                np.testing.assert_allclose(np.linalg.norm(pred-a['truth'],axis=1).mean(),float(row['ADE_10s_m']),rtol=1e-12)
assert len(list(root.glob('*.png')))==13
result={'status':'passed','matched_forecast_windows':12,'history_conditions':60,'method_results':300,'same_future_verified':True,'nested_identical_observations_verified':True,'all_metrics_recomputed':True,'png_count':13}
(root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
