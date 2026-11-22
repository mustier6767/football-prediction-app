import unittest
from soccer import model
import pandas as pd
class Tests(unittest.TestCase):
    def test_probabilities(self):
        dates=pd.date_range('2026-01-01',periods=60)
        df=pd.DataFrame({'date':dates,'HomeTeam':['1']*60,'AwayTeam':['2']*60,'FTHG':[2]*60,'FTAG':[1]*60})
        p=model(df,'1','2')
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p['home_win']+p['draw']+p['away_win'],1,places=8)
        self.assertAlmostEqual(p['btts_yes']+p['btts_no'],1,places=8)
    def test_minimum_history(self):
        df=pd.DataFrame({'date':[],'HomeTeam':[],'AwayTeam':[],'FTHG':[],'FTAG':[]})
        self.assertIsNone(model(df,'1','2'))
if __name__=='__main__':unittest.main()
