# MatchLab — simple iPhone dashboard

Displays today's and the next seven calendar days' scheduled matches for the Premier League, La Liga, Serie A, Bundesliga and Ligue 1. Fixtures and current-season completed matches are fetched automatically from football-data.org. No CSV uploads.

## One-time setup

1. Register for a **free API token** at https://www.football-data.org/client/register . Check email for the token.
2. In your existing GitHub repository, **replace `app.py`, `soccer.py`, and `requirements.txt`** with the files in this ZIP (at repository root). Keep other files if desired.
3. Open https://share.streamlit.io/ and open your deployed app's **Settings → Secrets**. Paste exactly:
   ```toml
   FOOTBALL_DATA_TOKEN = "YOUR_REAL_TOKEN"
   ```
   Save and reboot/redeploy the app. **Do not put the token in GitHub** or screenshots.
4. In Safari, open your Streamlit app link. Choose All five leagues or a specific league. Predictions appear automatically. Add to Home Screen via Safari Share menu.

## How predictions work

The baseline `soccer.model` is a recency-weighted Poisson model with league and team goal rates, shrinking sparse teams toward league averages. Uses *only finished current-season matches* before each fixture date; minimum 50 league matches. Shows home/draw/away, over/under, BTTS, expected goals and double chance. If the model lacks enough matches, the app says so instead of inventing numbers. It is not a trained AI neural network and its predictions are not validated to be profitable.

## Data limitations

- Free football-data.org access includes these five leagues, but fixtures/scores can be delayed and API availability/rate limits apply. 10 calls/minute free plan; the app caches responses (1h fixtures, 6h histories).
- The app displays upcoming scheduled games, not live scores. Dates are selected in UTC; display times are UK time, so a fixture close to midnight may fall on a different local calendar day.
- The API's free plan may only provide the current season's completed games. New seasons will initially lack enough data. More history requires an appropriate data subscription or another licensed provider.
- No live odds, no EV recommendation, no automated bet placement. Historical backtesting remains available in your **previous** app/project; this simplified interface focuses on automatic forecasts.
- Paper/research use only. Football predictions can be wrong; earlier MatchLab backtest was negative.

## Run locally

```bash
pip install -r requirements.txt
export FOOTBALL_DATA_TOKEN='your-token'
streamlit run app.py
```

## Quick tests

```bash
python -m unittest test_app.py
```
