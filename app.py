"""MatchLab: automatic seven-day soccer fixtures and paper-only model forecasts."""
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from soccer import model

st.set_page_config(page_title="MatchLab | Next 7 Days", page_icon="⚽", layout="centered")
st.markdown('''<style>.block-container{max-width:800px;padding:1rem .85rem 4rem}h1{font-size:1.8rem!important}.stButton button{min-height:45px;border-radius:12px}[data-testid="stMetric"]{background:rgba(130,130,130,.10);padding:10px;border-radius:12px}</style>''', unsafe_allow_html=True)
LEAGUES = {'PL':'Premier League','PD':'La Liga','SA':'Serie A','BL1':'Bundesliga','FL1':'Ligue 1'}
BASE='https://api.football-data.org/v4'

def token():
    try: return st.secrets.get('FOOTBALL_DATA_TOKEN', os.environ.get('FOOTBALL_DATA_TOKEN',''))
    except Exception: return os.environ.get('FOOTBALL_DATA_TOKEN','')

@st.cache_data(ttl=3600,show_spinner=False)
def fetch(path, params, api_token):
    response=requests.get(BASE+path,params=params,headers={'X-Auth-Token':api_token},timeout=25)
    if response.status_code!=200:
        try: msg=response.json().get('message','')
        except ValueError: msg=''
        raise RuntimeError(f'Football API returned HTTP {response.status_code}. {msg}')
    return response.json()

def completed_frame(matches):
    rows=[]
    for m in matches:
        score=m.get('score') or {}
        ft=score.get('fullTime') or {}
        h=ft.get('home'); a=ft.get('away')
        if m.get('status')!='FINISHED' or h is None or a is None: continue
        home=m.get('homeTeam') or {};away=m.get('awayTeam') or {}
        if home.get('id') is None or away.get('id') is None: continue
        dt=pd.to_datetime(m.get('utcDate'),utc=True,errors='coerce')
        if pd.isna(dt):continue
        rows.append({'date':dt.tz_localize(None).normalize(), 'HomeTeam':str(home['id']),
                     'AwayTeam':str(away['id']),'FTHG':int(h),'FTAG':int(a)})
    return pd.DataFrame(rows,columns=['date','HomeTeam','AwayTeam','FTHG','FTAG'])

def forecast(history,match):
    h=match.get('homeTeam') or {};a=match.get('awayTeam') or {}
    if h.get('id') is None or a.get('id') is None:return None
    # The model uses completed league games only, strictly before this fixture date.
    date=pd.to_datetime(match['utcDate'],utc=True).tz_localize(None).normalize()
    hist=history[history.date<date]
    return model(hist,str(h['id']),str(a['id']))

def percent(x):return f'{x*100:.0f}%'

def match_title(m):
    h=m.get('homeTeam') or {};a=m.get('awayTeam') or {}
    return f"{h.get('shortName') or h.get('name') or 'TBD'}  vs  {a.get('shortName') or a.get('name') or 'TBD'}"

st.title('⚽ MatchLab')
st.caption('Today + next 7 days · Five leagues · Automatic fixtures · Paper-only forecasts')
st.info('Model estimates, not guaranteed outcomes. No bookmaker connection or real-money betting.')
api_token=token()
if not api_token:
    st.warning('One-time setup needed: add your free football-data.org API token in Streamlit app secrets. See README.')
    st.code('FOOTBALL_DATA_TOKEN = "paste-your-token-here"',language='toml')
    st.stop()

now=datetime.now(timezone.utc)
start=now.date();end=start+timedelta(days=7)
league_choice=st.selectbox('League', ['All five leagues']+list(LEAGUES.values()))
selected_codes=[c for c,n in LEAGUES.items() if league_choice=='All five leagues' or n==league_choice]
try:
    # One request for upcoming matches, five for league history (cached for 6 hours below).
    upcoming=fetch('/matches',{'competitions':','.join(selected_codes),'dateFrom':start.isoformat(),
                                'dateTo':end.isoformat()},api_token).get('matches',[])
except (requests.RequestException,RuntimeError,ValueError) as exc:
    st.error(f'Could not load fixtures: {exc}');st.stop()

upcoming=[m for m in upcoming if m.get('status') in ('SCHEDULED','TIMED') and
          m.get('competition',{}).get('code') in selected_codes and
          pd.to_datetime(m['utcDate'],utc=True).to_pydatetime()>=now]
upcoming.sort(key=lambda m:m['utcDate'])
st.caption(f'{len(upcoming)} upcoming matches found · {start:%d %b}–{end:%d %b %Y} · times in UK time')
if st.button('🔄 Refresh fixtures'):
    fetch.clear();st.rerun()
if not upcoming:
    st.info('No upcoming fixtures found for this league in the next seven days. Check again later.')
    st.stop()

# Fetch only leagues actually needed, cache league histories across user sessions.
@st.cache_data(ttl=21600,show_spinner=False)
def history_for_league(code,api_token):
    data=fetch(f'/competitions/{code}/matches',{},api_token)
    return completed_frame(data.get('matches',[]))

histories={};failures={}
for code in sorted({m['competition']['code'] for m in upcoming}):
    try: histories[code]=history_for_league(code,api_token)
    except (requests.RequestException,RuntimeError,ValueError) as exc:failures[code]=str(exc)

for m in upcoming:
    code=m['competition']['code']; dt=pd.to_datetime(m['utcDate'],utc=True).tz_convert(ZoneInfo('Europe/London'))
    with st.container(border=True):
        st.caption(f"{LEAGUES.get(code,code)} · {dt:%a %d %b, %H:%M} UK")
        st.subheader(match_title(m))
        if code in failures:
            st.warning('Could not load historical league data: '+failures[code]);continue
        p=forecast(histories[code],m)
        if p is None:
            st.warning('Not enough completed matches to produce a supported prediction yet (minimum 50 in this league).')
            continue
        c1,c2,c3=st.columns(3)
        c1.metric('Home win',percent(p['home_win']))
        c2.metric('Draw',percent(p['draw']))
        c3.metric('Away win',percent(p['away_win']))
        a,b=st.columns(2)
        a.metric('Over 2.5',percent(p['over_2_5']))
        b.metric('BTTS Yes',percent(p['btts_yes']))
        st.caption(f"Expected goals: {p['expected_home_goals']:.2f} – {p['expected_away_goals']:.2f} · Model trained on {len(histories[code])} completed current-season league games")
        with st.expander('More markets'):
            st.write(f"Over 1.5: {percent(p['over_1_5'])} · Under 2.5: {percent(p['under_2_5'])} · Over 3.5: {percent(p['over_3_5'])}")
            st.write(f"BTTS No: {percent(p['btts_no'])} · Double chance 1X: {percent(p['double_1x'])} · X2: {percent(p['double_x2'])}")

st.divider()
st.caption('Source: football-data.org (fixtures and completed results). The free plan can delay schedules/scores. This baseline is not independently validated for profitable betting; odds and EV are intentionally excluded without a licensed, time-stamped odds feed.')
