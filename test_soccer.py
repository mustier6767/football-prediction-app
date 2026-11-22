import numpy as np
import pandas as pd
from soccer import model,outcomes,metrics

def test_probabilities():
    dates=pd.date_range('2022-01-01',periods=120,freq='D')
    df=pd.DataFrame({'date':dates,'HomeTeam':['A','B']*60,'AwayTeam':['B','A']*60,
                     'FTHG':[2,1]*60,'FTAG':[1,1]*60})
    p=model(df,'A','B')
    assert abs(p['home_win']+p['draw']+p['away_win']-1)<1e-10
    assert abs(p['btts_yes']+p['btts_no']-1)<1e-10
    assert abs(p['over_2_5']+p['under_2_5']-1)<1e-10
    assert 0<=p['home_win']<=1

def test_outcomes():
    o=outcomes(2,1)
    assert o['H'] and o['O25'] and o['Y'] and o['1X']
    assert not o['D'] and not o['U25']

def test_empty_metrics():
    assert metrics(pd.DataFrame(),pd.DataFrame())['bets']==0
