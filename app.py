"""Mobile-first, paper-only Streamlit interface for the five-league Poisson model."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from soccer import LEAGUES, MARKETS, metrics, model, prepare, run_backtest

st.set_page_config(page_title='MatchLab | Paper Predictions', page_icon='⚽', layout='centered', initial_sidebar_state='collapsed')
st.markdown('''<style>
.block-container{max-width:850px;padding-top:1.1rem;padding-bottom:4rem}
h1{font-size:2rem!important;letter-spacing:-.04em}h2{font-size:1.35rem!important}
[data-testid="stMetric"]{background:rgba(120,120,120,.075);padding:12px;border-radius:14px}
[data-testid="stMetricValue"]{font-size:1.35rem}
.stButton>button{border-radius:12px;min-height:44px}
@media(max-width:600px){.block-container{padding-left:.85rem;padding-right:.85rem}h1{font-size:1.75rem!important}}
</style>''', unsafe_allow_html=True)

st.title('⚽ MatchLab')
st.caption('Five-league statistical forecasts · Historical simulations · Paper-only')
st.info('Research dashboard only. No real-money betting, accounts, or bookmaker connections. Forecasts are estimates, not guarantees.', icon='🧪')

@st.cache_data(show_spinner=False)
def load_local(folder):
    return prepare(folder)

@st.cache_data(show_spinner=False)
def load_uploaded(files):
    with tempfile.TemporaryDirectory() as folder:
        for name, content in files:
            Path(folder, Path(name).name).write_bytes(content)
        return prepare(folder)

with st.expander('📂 Data source and instructions', expanded=False):
    st.write('Upload historical Football-Data-style CSV files with Date, Div, HomeTeam, AwayTeam, FTHG, FTAG. Optional odds: B365H/D/A and B365>2.5 / B365<2.5 (or Avg equivalents).')
    st.write('Or put CSVs in a `data/` folder before deploying. Only completed historical matches are used for fitting. Upload only data you are permitted to use.')
    st.caption('The data provider may restrict automated or commercial use. This app does not automatically fetch or place bets.')

uploads=st.file_uploader('Historical match CSV files', type=['csv'], accept_multiple_files=True, help='Upload one or more CSVs for the five supported leagues.')
try:
    if uploads:
        dataset=load_uploaded(tuple((f.name,f.getvalue()) for f in uploads))
        source='Uploaded CSV files'
    elif Path('data').exists() and any(Path('data').glob('*.csv')):
        dataset=load_local('data')
        source='Bundled data/ CSV files'
    else:
        dataset=None
        source=None
except Exception as exc:
    dataset=None
    st.error(f'Could not read match data: {exc}')

if dataset is not None:
    st.success(f'{len(dataset):,} completed matches loaded · {source}')
    st.caption(f'Coverage: {dataset.date.min().date()} – {dataset.date.max().date()}')
else:
    st.warning('No historical data loaded. Upload CSV files above to activate predictions and backtesting.')

tab_predict,tab_backtest,tab_ev=st.tabs(['🔮 Predict', '📈 Backtest', '🧮 EV calculator'])

with tab_predict:
    st.subheader('Match prediction')
    if dataset is not None:
        league=st.selectbox('League', list(LEAGUES), format_func=lambda x:LEAGUES[x], key='pred_league')
        league_data=dataset[dataset.Div==league].sort_values('date')
        # Use the most recent two years; never pretend to have lineup or live data.
        latest=league_data.date.max()
        history=league_data[league_data.date>=latest-pd.Timedelta(days=730)]
        teams=sorted(set(history.HomeTeam).union(history.AwayTeam))
        if len(teams)<2 or len(history)<50:
            st.warning('Not enough historical data for this league.')
        else:
            c1,c2=st.columns(2)
            with c1: home=st.selectbox('Home team',teams,index=0)
            with c2: away=st.selectbox('Away team',teams,index=min(1,len(teams)-1))
            if home==away:
                st.warning('Select two different teams.')
            else:
                p=model(history,home,away)
                if p is None:st.warning('At least 50 league matches are required.')
                else:
                    st.caption(f'Model trained through {latest.date()} · Not a live fixture or injury feed')
                    c1,c2,c3=st.columns(3)
                    c1.metric('Home',f'{p["home_win"]:.1%}')
                    c2.metric('Draw',f'{p["draw"]:.1%}')
                    c3.metric('Away',f'{p["away_win"]:.1%}')
                    st.progress(float(p['home_win']),text='Home win probability')
                    st.write(f'**Expected goals:** {home} {p["expected_home_goals"]:.2f} — {p["expected_away_goals"]:.2f} {away}')
                    st.dataframe(pd.DataFrame([
                        {'Market':'Over 1.5','Probability':p['over_1_5']},
                        {'Market':'Over 2.5','Probability':p['over_2_5']},
                        {'Market':'Under 2.5','Probability':p['under_2_5']},
                        {'Market':'Over 3.5','Probability':p['over_3_5']},
                        {'Market':'BTTS Yes','Probability':p['btts_yes']},
                        {'Market':'BTTS No','Probability':p['btts_no']},
                        {'Market':'Double chance 1X','Probability':p['double_1x']},
                        {'Market':'Double chance X2','Probability':p['double_x2']},
                        {'Market':'Double chance 12','Probability':p['double_12']},
                    ]).style.format({'Probability':'{:.1%}'}),hide_index=True,use_container_width=True)
                    st.caption('Teams are selected manually; this app does not supply a verified upcoming fixture list.')
    else:st.caption('Load historical data to enable match predictions.')

with tab_backtest:
    st.subheader('Historical walk-forward backtest')
    if dataset is not None:
        league_choice=st.selectbox('Leagues', ['ALL']+list(LEAGUES),format_func=lambda x:'All five leagues' if x=='ALL' else LEAGUES[x])
        earliest=dataset.date.min().date();latest=dataset.date.max().date()
        suggested=max(earliest,min(latest, pd.Timestamp('2023-08-01').date()))
        start=st.date_input('First out-of-sample test date',value=suggested,min_value=earliest,max_value=latest)
        c1,c2=st.columns(2)
        with c1:edge=st.slider('Minimum probability edge',0.0,0.20,0.04,0.01)
        with c2:min_ev=st.slider('Minimum expected ROI',0.0,0.30,0.03,0.01)
        stake=st.number_input('Paper stake per selection (units)',min_value=0.1,max_value=10000.0,value=1.0,step=1.0)
        st.caption('Same-day matches are predicted together using earlier dates only. Historical CSV odds may not represent executable pre-match prices. Thresholds must be selected on validation data, not tuned to the final test period.')
        if st.button('Run paper backtest',type='primary',use_container_width=True):
            subset=dataset if league_choice=='ALL' else dataset[dataset.Div==league_choice]
            with st.spinner('Computing chronological forecasts…'):
                forecasts,bets=run_backtest(subset,str(start),min_edge=edge,min_ev=min_ev,stake=stake)
                summary=metrics(forecasts,bets)
            st.session_state['backtest']=(forecasts,bets,summary)
        if 'backtest' in st.session_state:
            forecasts,bets,summary=st.session_state['backtest']
            a,b,c=st.columns(3)
            a.metric('Matches',f'{summary["matches_forecast"]:,}')
            b.metric('Paper selections',f'{summary["bets"]:,}')
            c.metric('ROI',f'{summary.get("roi",0):+.1%}' if len(bets) else 'N/A')
            if len(bets):
                d,e,f=st.columns(3)
                d.metric('Net units',f'{summary["net_profit"]:+.2f}')
                e.metric('Win rate',f'{summary["win_rate"]:.1%}')
                f.metric('Max drawdown',f'{summary["max_drawdown_units"]:.2f} units')
                equity=bets.groupby('date').profit.sum().sort_index().cumsum()
                st.line_chart(equity.rename('Cumulative paper profit'),use_container_width=True)
                st.dataframe(bets.sort_values('date',ascending=False).head(100),hide_index=True,use_container_width=True)
            else:st.info('No selections met the EV thresholds or historical odds were unavailable. Predictions can still be evaluated.')
            if 'brier_1x2' in summary:
                st.write(f'**1X2 Brier:** {summary["brier_1x2"]:.4f} · **Log loss:** {summary["log_loss_1x2"]:.4f}')
            st.download_button('Download forecasts.csv',forecasts.to_csv(index=False),'forecasts.csv','text/csv',use_container_width=True)
            st.download_button('Download paper_bets.csv',bets.to_csv(index=False),'paper_bets.csv','text/csv',use_container_width=True)
            st.download_button('Download summary.json',json.dumps(summary,indent=2),'summary.json','application/json',use_container_width=True)
    else:st.caption('Load historical data to enable walk-forward backtesting.')

with tab_ev:
    st.subheader('Expected value — hypothetical only')
    probability=st.slider('Your estimated outcome probability',1,99,55,1)/100
    odds=st.number_input('Decimal odds',min_value=1.01,max_value=1000.0,value=2.00,step=0.05)
    paper_stake=st.number_input('Hypothetical stake (units)',min_value=0.1,max_value=10000.0,value=10.0,step=1.0)
    expected_roi=probability*odds-1
    a,b=st.columns(2)
    a.metric('Break-even probability',f'{1/odds:.1%}')
    b.metric('Expected ROI',f'{expected_roi:+.1%}')
    st.metric('Expected profit per paper selection',f'{paper_stake*expected_roi:+.2f} units')
    st.caption('A positive calculated EV does not establish a real edge. It depends entirely on the quality and calibration of the probability estimate.')

st.divider()
st.caption('MatchLab · Research and education · No live odds, no guaranteed returns, no real-money execution')
