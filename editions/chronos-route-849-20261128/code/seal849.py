"""Seal every E1 issuance against its full paid observation and native inference identity."""
from chronos_route_heads import *
from prefix849 import modelid,extend
from grudata import schedule
import gzip,io
def seal(job):
    split,k=job;w,wm=world(split,k);dest=OUT/f'E1/prefix_index/{split}/{k}'
    src=dest/'prefixes.pkl';r=get(dest/'receipt.json');out=dest/'canonical-full-prefixes.json.gz';rec=dest/'canonical-receipt.json'
    retrieval=dest/'retrieval_features.npz'
    assert r['physics']==wm['physical_sha256'] and sha_file(src)==r['sha256']
    if rec.exists():
        z=get(rec);assert z['source_sha']==r['sha256'] and z['sha']==sha_file(out) and z['retrieval_sha']==sha_file(retrieval);return k
    archive=pickle.loads(src.read_bytes());old=[EMPTY]*16;full=[EMPTY]*16;count=[0]*16;keys=[];ids=set();memories=[Memory() for _ in range(16)];summaries=[];supports=[];causal=[]
    for row,(u,t) in zip(archive,schedule(w,41000+k),strict=True):
        uid,o,n,old_sha,csha,c,width=row;physical=w.observe(u,t)
        assert uid==u and obsfull(physical)==obsfull(o) and csha==arrhash(c)
        old[u]=extend(old[u],o);assert old[u]==old_sha
        full[u]=digest(full[u],o);count[u]+=1;assert n==count[u]
        key=[split,w.namespace,wm['physical_sha256'],u,n,t,full[u],modelid(),pf.GRID.tolist(),str(c.dtype),str(width.dtype)]
        ksha=sha(compact(key));assert ksha not in ids;ids.add(ksha)
        keys.append([ksha,key,csha,arrhash(width)])
        mem=memories[u];mem.on_paid_fix(o);q=mem.query();query_start=mem.h[-3].t if len(mem.h)>=3 else -1.
        eligible=[r for r in mem.records if r[0]<query_start and r[1]<query_start]
        assert q[3]==min(3,len(eligible)) and q[4]<=q[3] and (not q[5] or len(eligible)>=2)
        summaries.append(np.stack(q[:3]));supports.append(q[3:]);causal.append([query_start,len(eligible),max((v[0] for v in eligible),default=-1.),max((v[1] for v in eligible),default=-1.)])
        mem.issue(c)
    buffer=io.BytesIO();np.savez_compressed(buffer,features=np.stack(summaries),support=np.asarray(supports),eligibility=np.asarray(causal),full_key_sha=np.asarray([v[0] for v in keys]))
    atom(retrieval,buffer.getvalue())
    atom(out,gzip.compress(compact(keys),compresslevel=4,mtime=0))
    put(rec,dict(world=k,physics=wm['physical_sha256'],source_sha=r['sha256'],sha=sha_file(out),retrieval_sha=sha_file(retrieval),retrieval_axes=['TIME','ROUTE','SHUFFLE'],retrieval_fields=['mean_abs_az','mean_abs_el','max_abs_az','max_abs_el','mean_feedback_age','mean_measurement_cov_trace','mean_match_distance','mean_one_minus_heading_cosine','neighbors','eligible_records'],complete_field_prefixes=len(keys),duplicate_keys=0,source_fields_checked=['t','position','covariance','radial_velocity','radial_std','raw'],schema=['split','namespace','physics','uid','paid_count','time','complete_prefix','model','grid','center_dtype','width_dtype']))
    return k
