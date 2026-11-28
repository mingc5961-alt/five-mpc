"""Fetch only original v1 camera-time motion-capture tables, checking the ZIP member CRCs."""
import bisect
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import struct
import time
import zipfile
import zlib
import pandas as pd
import requests
from server_paths import RUN

REPO = 'tii-racing/drone-racing-dataset'
VERSION = 'v1.0.0'
OUT = RUN/'inputs/bosello'


class RemoteParts(io.RawIOBase):
    def __init__(self, parts):
        self.parts = parts
        self.ends = [0]
        for part in parts:
            self.ends.append(self.ends[-1]+part['size'])
        self.offset = 0

    def seek(self, n, whence=0):
        base = (0,self.offset,self.ends[-1])[whence]
        if n+base < 0:
            raise OSError('negative seek')
        self.offset=n+base
        return self.offset

    def tell(self):
        return self.offset

    def read(self, n=-1):
        end=self.ends[-1] if n<0 else min(self.offset+n,self.ends[-1])
        buf=bytearray()
        while self.offset<end:
            i=bisect.bisect_right(self.ends,self.offset)-1
            lo=self.offset-self.ends[i]
            hi=min(end,self.ends[i+1])-self.ends[i]-1
            url=self.parts[i]['browser_download_url']
            for attempt in range(4):
                try:
                    r=requests.get(url,params={'g1_slice':f'{lo}-{hi}'},headers={'Range':f'bytes={lo}-{hi}'},timeout=60)
                    r.raise_for_status()
                    if r.status_code!=206 or len(r.content)!=hi-lo+1 or not r.headers.get('Content-Range','').startswith(f'bytes {lo}-{hi}/'):
                        raise IOError('incorrect HTTP range')
                    break
                except (requests.RequestException,IOError):
                    if attempt==3: raise
                    time.sleep(1+attempt)
            buf.extend(r.content)
            self.offset+=hi-lo+1
        return bytes(buf)


def fetch(args):
    parts,member=args
    stream=RemoteParts(parts)
    stream.seek(member.header_offset)
    payload=stream.read(member.compress_size+2048)
    header=struct.unpack('<4s5H3I2H',payload[:30])
    assert header[0]==b'PK\x03\x04'
    start=30+header[-1]+header[-2]
    assert not header[2]&1 and start<=2048
    raw=payload[start:start+member.compress_size]
    data=zlib.decompress(raw,-15) if member.compress_type==zipfile.ZIP_DEFLATED else raw
    if zlib.crc32(data)!=member.CRC or len(data)!=member.file_size:
        raise IOError(member.filename)
    fields=['elapsed_time','drone_x','drone_y','drone_z']
    table=pd.read_csv(io.BytesIO(data),usecols=fields)
    dest=OUT/(member.filename.split('/')[-1]+'.gz')
    table.to_csv(dest,index=False,compression='gzip')
    return dict(group=parts[0]['name'].split('_')[0],member=member.filename,local=dest.name,
                source_sha256=hashlib.sha256(data).hexdigest(),zip_crc32=f'{member.CRC:08x}',
                samples=len(table),columns=fields)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    rel=requests.get(f'https://api.github.com/repos/{REPO}/releases/tags/{VERSION}',timeout=20)
    rel.raise_for_status()
    assets=rel.json()['assets'];jobs=[]
    for group in ('piloted','autonomous'):
        parts=sorted([a for a in assets if a['name'].startswith(group+'_zipchunk')],key=lambda a:a['name'])
        with zipfile.ZipFile(RemoteParts(parts)) as archive:
            entries=[i for i in archive.infolist() if i.filename.endswith('_cam_ts_sync.csv') and '__MACOSX' not in i.filename]
            print(group,len(entries),[i.filename for i in entries[:2]],flush=True)
            jobs.extend((parts,m) for m in entries)
    if not jobs:raise ValueError('No archive members selected')
    with ThreadPoolExecutor(6) as pool:files=list(pool.map(fetch,jobs))
    (OUT/'source.json').write_text(json.dumps(dict(release=rel.json()['html_url'],files=files),indent=2))
