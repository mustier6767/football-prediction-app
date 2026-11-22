"""Five-league paper-only soccer forecasts and chronological backtests."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

LEAGUES = {'E0':'Premier League','SP1':'La Liga','I1':'Serie A','D1':'Bundesliga','F1':'Ligue 1'}
MARKETS = {'H':'home_win','D':'draw','A':'away_win','O15':'over_1_5','U15':'under_1_5',
           'O25':'over_2_5','U25':'under_2_5','O35':'over_3_5','U35':'under_3_5',
           'Y':'btts_yes','N':'btts_no','1X':'double_1x','X2':'double_x2','12':'double_12'}
ODD_COLS = {'H':['B365H','AvgH'],'D':['B365D','AvgD'],'A':['B365A','AvgA'],
             'O25':['B365>2.5','Avg>2.5'],'U25':['B365<2.5','Avg<2.5']}

def prepare(folder):
    frames=[]
    for path in sorted(Path(folder).glob('*.csv')):
        try:
            d=pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip', low_memory=False)
        except (UnicodeError, pd.errors.ParserError):
            d=pd.read_csv(path, encoding='latin1', on_bad_lines='skip', low_memory=False)
        d.columns=d.columns.str.strip()
        if not {'Date','HomeTeam','AwayTeam','FTHG','FTAG'}.issubset(d.columns):
            continue
        if 'Div' not in d: continue
        d=d[d.Div.isin(LEAGUES)].copy()
        if d.empty: continue
        d['date']=pd.to_datetime(d.Date, dayfirst=True, errors='coerce')
        for c in ['FTHG','FTAG']:
            d[c]=pd.to_numeric(d[c],errors='coerce')
        d=d.dropna(subset=['date','HomeTeam','AwayTeam','FTHG','FTAG'])
        d=d[(d.FTHG>=0)&(d.FTAG>=0)]
        d['source_file']=path.name
        frames.append(d)
    if not frames: raise ValueError('No valid historical CSVs found. See README.')
    out=pd.concat(frames,ignore_index=True)
    out=out.sort_values(['date','Div','HomeTeam','AwayTeam','source_file'])
    out=out.drop_duplicates(['date','Div','HomeTeam','AwayTeam'],keep='first').reset_index(drop=True)
    return out

def model(history, home, away, half_life=365, prior=8, max_goals=12):
    """Exponentially weighted league/team goal rates, with shrinkage to league baseline."""
    if len(history)<50: return None
    h=history.copy()
    days=(h.date.max()-h.date).dt.days.clip(lower=0).to_numpy()
    weights=np.power(0.5, days/half_life)
    hgo=h.FTHG.to_numpy(dtype=float); ago=h.FTAG.to_numpy(dtype=float)
    base_h=np.average(hgo,weights=weights); base_a=np.average(ago,weights=weights)
    if min(base_h,base_a)<=0: return None
    def avg(mask, values, base):
        w=weights[mask]; v=values[mask]
        return (np.dot(w,v)+prior*base)/(w.sum()+prior)
    home_mask=(h.HomeTeam==home).to_numpy()
    away_mask=(h.AwayTeam==away).to_numpy()
    # A promoted/new team gets a league-average prior rather than future data.
    attack_h=avg(home_mask,hgo,base_h)/base_h
    defense_a=avg(away_mask,hgo,base_h)/base_h
    attack_a=avg(away_mask,ago,base_a)/base_a
    defense_h=avg(home_mask,ago,base_a)/base_a
    lam_h=np.clip(base_h*attack_h*defense_a,0.15,5.0)
    lam_a=np.clip(base_a*attack_a*defense_h,0.15,5.0)
    # Tail included in last bucket: probabilities sum exactly to 1.
    def pmf(lam):
        p=poisson.pmf(np.arange(max_goals),lam)
        return np.r_[p,max(0.,1-p.sum())]
    mat=np.outer(pmf(lam_h),pmf(lam_a))
    i,j=np.indices(mat.shape); total=i+j
    pH=float(mat[i>j].sum()); pD=float(mat[i==j].sum()); pA=float(mat[i<j].sum())
    btts=float(mat[(i>0)&(j>0)].sum())
    out={'expected_home_goals':float(lam_h),'expected_away_goals':float(lam_a),
         'home_win':pH,'draw':pD,'away_win':pA,'btts_yes':btts,'btts_no':1-btts,
         'double_1x':pH+pD,'double_x2':pD+pA,'double_12':pH+pA}
    for threshold in (1,2,3):
        p=float(mat[total>threshold].sum())
        out[f'over_{threshold}_5']=p;out[f'under_{threshold}_5']=1-p
    return out

def outcomes(h,a):
    h=int(h);a=int(a);t=h+a
    return {'H':h>a,'D':h==a,'A':h<a,'O15':t>1,'U15':t<=1,
            'O25':t>2,'U25':t<=2,'O35':t>3,'U35':t<=3,
            'Y':h>0 and a>0,'N':h==0 or a==0,'1X':h>=a,'X2':h<=a,'12':h!=a}

def price(row, market):
    for c in ODD_COLS.get(market,[]):
        if c in row.index:
            x=pd.to_numeric(row[c],errors='coerce')
            if pd.notna(x) and np.isfinite(x) and 1.01<float(x)<1000:
                return float(x),c
    return np.nan,''

def run_backtest(df, start, min_edge=0.04, min_ev=0.03, stake=1.0,
                 train_days=730, half_life=365, prior=8, odds_mode='historical'):
    """Predict entire same-date batch using only matches on earlier dates."""
    forecasts=[]; bets=[]
    for league,league_df in df.groupby('Div'):
        league_df=league_df.sort_values('date')
        for date,day in league_df.groupby('date',sort=True):
            if date < pd.Timestamp(start): continue
            history=league_df[(league_df.date<date)&(league_df.date>=date-pd.Timedelta(days=train_days))]
            if len(history)<50: continue
            for _,row in day.iterrows():
                p=model(history,row.HomeTeam,row.AwayTeam,half_life=half_life,prior=prior)
                if p is None: continue
                actual=outcomes(row.FTHG,row.FTAG)
                record={'date':date,'league':league,'home':row.HomeTeam,'away':row.AwayTeam,
                        'home_goals':int(row.FTHG),'away_goals':int(row.FTAG),**p}
                forecasts.append(record)
                if odds_mode=='none':continue
                for code,key in MARKETS.items():
                    odds,source=price(row,code)
                    if not np.isfinite(odds):continue
                    prob=p[key]; breakeven=1/odds; ev=prob*odds-1
                    if (prob-breakeven)<min_edge or ev<min_ev:continue
                    won=bool(actual[code]); profit=stake*(odds-1) if won else -stake
                    bets.append({'date':date,'league':league,'home':row.HomeTeam,'away':row.AwayTeam,
                                 'market':code,'probability':prob,'odds':odds,'odds_column':source,
                                 'edge':prob-breakeven,'expected_roi':ev,'won':won,
                                 'stake':stake,'profit':profit})
    return pd.DataFrame(forecasts),pd.DataFrame(bets)

def metrics(forecasts,bets):
    result={'matches_forecast':len(forecasts),'bets':len(bets)}
    if len(forecasts):
        actual=np.select([forecasts.home_goals>forecasts.away_goals,
                          forecasts.home_goals==forecasts.away_goals],[0,1],default=2)
        probs=forecasts[['home_win','draw','away_win']].to_numpy()
        y=np.eye(3)[actual]
        result['brier_1x2']=float(np.mean(np.sum((probs-y)**2,axis=1)))
        result['log_loss_1x2']=float(-np.mean(np.log(np.maximum(probs[np.arange(len(probs)),actual],1e-12))))
    if len(bets):
        total=float(bets.stake.sum()); net=float(bets.profit.sum())
        # Chronological bankroll from zero, group same-date settlements together.
        daily=bets.groupby('date').profit.sum().sort_index().cumsum()
        peak=np.maximum.accumulate(np.r_[0.,daily.to_numpy()])
        drawdowns=peak[1:]-daily.to_numpy()
        result.update(total_staked=total,net_profit=net,roi=net/total,
                      win_rate=float(bets.won.mean()),max_drawdown_units=float(drawdowns.max()),
                      max_drawdown_pct_of_staked=float(drawdowns.max()/total))
    return result

def main():
    ap=argparse.ArgumentParser(description='Paper-only five-league soccer backtester')
    ap.add_argument('--data',default='data');ap.add_argument('--output',default='output')
    ap.add_argument('--start',default='2023-08-01',help='first date of out-of-sample backtest')
    ap.add_argument('--min-edge',type=float,default=.04)
    ap.add_argument('--min-ev',type=float,default=.03)
    ap.add_argument('--stake',type=float,default=1.)
    ap.add_argument('--train-days',type=int,default=730)
    ap.add_argument('--half-life',type=float,default=365.)
    ap.add_argument('--prior',type=float,default=8.)
    ap.add_argument('--odds-mode',choices=['historical','none'],default='historical')
    args=ap.parse_args()
    if args.stake<=0 or args.train_days<90 or args.half_life<=0 or args.prior<0:ap.error('invalid numeric arguments')
    data=prepare(args.data)
    forecasts,bets=run_backtest(data,args.start,args.min_edge,args.min_ev,args.stake,
                                args.train_days,args.half_life,args.prior,args.odds_mode)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    forecasts.to_csv(out/'forecasts.csv',index=False)
    bets.to_csv(out/'paper_bets.csv',index=False)
    import json
    summary=metrics(forecasts,bets)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if len(bets):
        b=bets.sort_values(['date','league','home','away','market']).copy()
        b['cumulative_profit']=b.profit.cumsum()
        b.to_csv(out/'equity.csv',index=False)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4))
        ax.plot(b.date,b.cumulative_profit);ax.axhline(0,color='gray',lw=.8)
        ax.set(xlabel='Match date',ylabel='Cumulative profit (units)',title='Paper-only historical backtest')
        fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/'equity.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))
    print('Files saved to',out.resolve())

if __name__=='__main__':main()
