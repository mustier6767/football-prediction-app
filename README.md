# MatchLab — iPhone-friendly five-league soccer model

A **paper-only** Streamlit web application using the original Poisson forecasting/backtesting engine. Supports Premier League (`E0`), La Liga (`SP1`), Serie A (`I1`), Bundesliga (`D1`), and Ligue 1 (`F1`).

## Use on iPhone (recommended: Streamlit Community Cloud)

1. On a computer (easiest), extract this ZIP and create a new GitHub repository. Upload the **contents** of this folder so `app.py`, `soccer.py`, and `requirements.txt` are at the repository root. GitHub's web interface can upload files. Do not upload confidential data.
2. Visit https://share.streamlit.io/ and sign in with GitHub. Choose **Create app**, select your repository, branch, and `app.py` as the entrypoint, then deploy. Hosting availability and policies can change.
3. Open the provided HTTPS app URL in **Safari on your iPhone**.
4. In Safari, tap **Share → Add to Home Screen → Add**. It will open like an app icon.
5. Upload historical CSVs using the app's **Historical match CSV files** uploader. The uploaded data are processed in the hosting environment. Avoid uploading anything sensitive or unlicensed.
6. Use **Predict** for manually selected team matchups, **Backtest** for historical simulations, and **EV calculator** for hypothetical odds calculations.

**Important:** This project does not include hosted infrastructure, a live fixture feed, live odds, bookmaker accounts, or a guarantee of profit. Streamlit sessions and uploaded data may reset on reload/restart; upload files again when necessary. A computer is strongly recommended for the initial GitHub deployment, though GitHub and Streamlit websites may be used from a phone.

## Run locally (computer)

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown in your terminal. For access from an iPhone on the same Wi-Fi network, use the computer's local network IP and Streamlit's port; configure firewall/network permissions appropriately. Do **not** expose your local app to the public internet without securing it.

## Data preparation

Provide historical Football-Data-format CSV files with required columns `Date`, `Div`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`. For odds comparisons, optional columns are `B365H`, `B365D`, `B365A`, `B365>2.5`, `B365<2.5` or `AvgH`, `AvgD`, `AvgA`, `Avg>2.5`, `Avg<2.5`. The app supports uploaded CSVs or a local `data/` folder.

If permitted by the provider's terms, the included optional downloader can fetch historical data on a computer:

```bash
python download_data.py --first 2018 --last 2025 --dest data
```

Check https://www.football-data.co.uk/ for terms and usage rights before automated downloading or redistribution. The app **does not automatically download** historical data.

## Model and testing

- For each test date, fit a league-specific recency-weighted Poisson model using the previous 730 days **strictly before** that date; games on the same date are forecast as a batch.
- Forecast 1X2, over/under 1.5/2.5/3.5, BTTS, double chance, and expected goals.
- Only 1X2 and over/under 2.5 have historical odds comparisons in the supplied engine, because these are the available supported odds fields. Other markets are forecast but **not** bet-simulated without corresponding odds.
- Historical odds in source CSVs may be closing prices, or may not be verifiably available before kickoff. Therefore simulated ROI is **not evidence of achievable live profit**.
- Report number of matches, Brier score, log loss, selection count, net units, ROI, win rate, and maximum drawdown in paper units. Zero selections produces no ROI.
- Choose model settings/EV thresholds on a **separate validation period**, freeze them, and evaluate on an untouched final test period. In-app threshold sliders are exploratory; tuning them repeatedly against the same test period invalidates an unbiased performance claim.
- No automated staking, no betting API, no financial transactions.

## Tests

```bash
python -m pytest -q
```

## Files

- `app.py` — mobile-responsive Streamlit interface
- `soccer.py` — original model and walk-forward backtest
- `download_data.py` — optional historical CSV downloader
- `test_soccer.py` — baseline unit tests
- `requirements.txt` — Python dependencies
