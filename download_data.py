"""Download public historical football-data.co.uk CSVs for personal research.
Review provider terms before use; avoid commercial/automated use without permission.
"""
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from soccer import LEAGUES


def download(first=2018,last=2025,dest='data'):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    for year in range(first,last+1):
        season=f'{year%100:02d}{(year+1)%100:02d}'
        for league in LEAGUES:
            url=f'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'
            path=dest/f'{season}_{league}.csv'
            try:
                req=Request(url,headers={'User-Agent':'Mozilla/5.0 (personal research)'})
                with urlopen(req,timeout=20) as r: data=r.read()
                if len(data)<100 or b'HomeTeam' not in data[:5000]:
                    raise ValueError('not a match CSV')
                path.write_bytes(data)
                print('OK',path,len(data))
            except Exception as e:
                print('SKIP',url,str(e))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--first',type=int,default=2018)
    p.add_argument('--last',type=int,default=2025)
    p.add_argument('--dest',default='data')
    a=p.parse_args()
    download(a.first,a.last,a.dest)
