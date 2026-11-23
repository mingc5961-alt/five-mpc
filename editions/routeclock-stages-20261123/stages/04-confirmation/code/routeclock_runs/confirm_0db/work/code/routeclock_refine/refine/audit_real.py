"""Build a file-level telemetry quality inventory before model scoring."""
import csv
import hashlib
import io
import sys
import zipfile
import numpy as np
import pandas as pd
from .data import ASSETS, BASE

def quality(t,xyz):
    keep=np.isfinite(t)&np.isfinite(xyz).all(1)
    times=np.unique(t[keep])
    if len(times)<2: return dict(valid_points=len(times))
    dt=np.diff(times)
    cuts=np.split(times,np.flatnonzero(dt>.5)+1)
    return dict(valid_points=len(times),raw_points=len(t),clock_reversals=int(np.sum(np.diff(t)<0)),
                dt_p50_s=np.median(dt),dt_p95_s=np.quantile(dt,.95),max_gap_s=max(dt),
                longest_contiguous_s=max(x[-1]-x[0] for x in cuts),span_s=times[-1]-times[0])

def main():
    out=BASE/'intake';out.mkdir(exist_ok=True)
    metadata=pd.read_csv(ASSETS/'cmu/parameters.csv')
    rows=[]
    with zipfile.ZipFile(ASSETS/'cmu/flights.zip') as z:
        for row in metadata.itertuples(index=False):
            data=z.read(f'flights/{row.flight}.csv');d=pd.read_csv(io.BytesIO(data))
            rows.append(dict(source='CMU',flight=row.flight,date=row.date,route=row.route,
                    status='route_candidate' if row.route.startswith('R') else 'hover_or_ground',
                    sha256=hashlib.sha256(data).hexdigest(),**quality(d.time.to_numpy(),d[['position_x','position_y','position_z']].to_numpy())))
    sys.path.insert(0,str(next((ASSETS/'wheels').glob('pyulog-*.whl'))))
    from pyulog import ULog
    for a in csv.DictReader((ASSETS/'px4/audit200.csv').open()):
        path=ASSETS/'px4'/(a['log_id']+'.ulg')
        result=dict(source='PX4',flight=a['log_id'],index_year=a['index_year'],url=a['url'])
        rows.append(result)
        if not path.exists():result['status']='pending_download';continue
        with path.open('rb') as f:result['sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        try:
            log=ULog(str(path),message_name_filter_list=['vehicle_global_position','vehicle_gps_position','sensor_gps','vehicle_status'])
            datasets={d.name:d.data for d in log.data_list if d.multi_id==0}
            hw=str(log.msg_info_dict.get('ver_hw','')).lower()
            if 'sitl' in hw or 'simulat' in hw or np.any(datasets.get('vehicle_status',{}).get('hil_state',0)):
                result['status']='sim_or_hil';continue
            for name in ('vehicle_global_position','vehicle_gps_position','sensor_gps'):
                if name not in datasets:continue
                d=datasets[name]
                if 'latitude_deg' in d:
                    xyz=np.column_stack([d['longitude_deg'],d['latitude_deg'],d['altitude_msl_m']])
                elif 'lat' in d:
                    xyz=np.column_stack([d['lon'],d['lat'],d['alt']]).astype(float)
                    if name!='vehicle_global_position':xyz*=np.array([1e-7,1e-7,1e-3])
                else:continue
                valid=(abs(xyz[:,0])<=180)&(abs(xyz[:,1])<=90)&(abs(xyz[:,:2]).sum(1)>0)
                if 'fix_type' in d:valid &= d['fix_type']>=3
                xyz[~valid]=np.nan
                q=quality(d['timestamp']*1e-6,xyz)
                if q['valid_points']<2:continue
                result.update(q,position_topic=name,status='position_candidate_flight_QC_pending')
                break
            else:result['status']='no_valid_global_position'
        except Exception as exc:
            result.update(status='parse_error',detail=str(exc))
    pd.DataFrame(rows).to_csv(out/'flight_manifest.csv',index=False)
    print(pd.DataFrame(rows).groupby(['source','status']).size().to_string())
if __name__=='__main__':main()
