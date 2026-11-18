"""
Gold (PKR/Tola) Forecast Dashboard

Setup:   pip install streamlit yfinance pandas numpy scikit-learn plotly
Run:     streamlit run gold_app.py

Educational tool, not financial advice.
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

OZ_G, TOLA_G = 31.1035, 11.6638  # grams per troy ounce / per tola
HORIZONS = [1, 7, 14, 30]        # trading days ahead

st.set_page_config(page_title="Gold Tola Forecast", layout="wide")
st.title("Gold Price Forecast (PKR / Tola)")

# ---------- Sidebar ----------
years = st.sidebar.slider("Years of history", 3, 15, 8)
premium = st.sidebar.slider("Local premium over international (%)", -5.0, 10.0, 0.0, 0.5)
band = st.sidebar.selectbox("Uncertainty band", ["80%", "90%"], index=0)
q_lo, q_hi = (0.10, 0.90) if band == "80%" else (0.05, 0.95)
st.sidebar.caption("Tola = 11.6638 g. Local bullion rates differ from the converted "
                   "international price; use the premium slider to calibrate.")


@st.cache_data(ttl=3600, show_spinner="Downloading data...")
def load(years):
    d = yf.download(["GC=F", "PKR=X"], period=f"{years}y",
                    auto_adjust=True, progress=False)["Close"].dropna()
    return d


def make_features(p, fx):
    r = np.log(p).diff()
    X = pd.DataFrame({
        "r1": r, "r5": r.rolling(5).sum(), "r21": r.rolling(21).sum(),
        "r63": r.rolling(63).sum(),
        "vol10": r.rolling(10).std(), "vol30": r.rolling(30).std(),
        "ma20": np.log(p / p.rolling(20).mean()),
        "ma100": np.log(p / p.rolling(100).mean()),
        "fx21": np.log(fx).diff(21),
    })
    return X


def new_model():
    return GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                     learning_rate=0.05, subsample=0.8, random_state=0)


@st.cache_data(show_spinner="Training models...")
def forecast(p, fx, h, q_lo, q_hi):
    X = make_features(p, fx)
    y = (np.log(p).shift(-h) - np.log(p)).rename("y")
    df = X.join(y).dropna()
    cols = list(X.columns)

    # Walk-forward backtest (gap = h avoids overlapping-target leakage)
    preds = pd.Series(np.nan, index=df.index)
    for tr, te in TimeSeriesSplit(5, gap=h).split(df):
        m = new_model().fit(df.iloc[tr][cols], df.iloc[tr]["y"])
        preds.iloc[te] = m.predict(df.iloc[te][cols])
    ev = preds.dropna()
    act = df["y"][ev.index]
    resid = act - ev

    final = new_model().fit(df[cols], df["y"])
    mu = final.predict(X.iloc[[-1]][cols])[0]
    lo, hi = resid.quantile([q_lo, q_hi])
    return {
        "mu": mu, "lo": mu + lo, "hi": mu + hi,
        "mae_model": (act - ev).abs().mean(),
        "mae_naive": act.abs().mean(),   # naive = "no change"
        "hit": (np.sign(ev) == np.sign(act)).mean(),
        "n": len(ev),
    }


try:
    data = load(years)
except Exception as e:
    st.error(f"Could not download data: {e}")
    st.stop()

usd_oz, fx = data["GC=F"], data["PKR=X"]
tola = usd_oz * fx * TOLA_G / OZ_G * (1 + premium / 100)
last, last_date = tola.iloc[-1], tola.index[-1]

# ---------- Forecasts ----------
res = {h: forecast(tola, fx, h, q_lo, q_hi) for h in HORIZONS}

c = st.columns(4)
c[0].metric("Current (est.)", f"Rs {last:,.0f}", f"as of {last_date:%d %b %Y}", delta_color="off")
for col, h in zip(c[1:], [1, 7, 30]):
    f = last * np.exp(res[h]["mu"])
    col.metric(f"Forecast +{h} trading day(s)", f"Rs {f:,.0f}", f"{f / last - 1:+.2%}")

# ---------- Chart ----------
hist = tola.iloc[-250:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History", line=dict(color="#c9a227")))
fx_dates = [last_date + BDay(h) for h in HORIZONS]
mid = [last * np.exp(res[h]["mu"]) for h in HORIZONS]
lo = [last * np.exp(res[h]["lo"]) for h in HORIZONS]
hi = [last * np.exp(res[h]["hi"]) for h in HORIZONS]
xs = [last_date] + fx_dates
fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[last] + hi + [last] + lo[::-1],
                         fill="toself", fillcolor="rgba(201,162,39,0.2)",
                         line=dict(width=0), name=f"{band} band", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xs, y=[last] + mid, name="Forecast", mode="lines+markers",
                         line=dict(color="#d9480f", dash="dash")))
fig.update_layout(height=450, yaxis_title="PKR per tola", margin=dict(t=20),
                  legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ---------- Backtest honesty table ----------
st.subheader("How reliable is the model? (walk-forward backtest)")
rows = []
for h in HORIZONS:
    r = res[h]
    rows.append({
        "Horizon (trading days)": h,
        "Forecast (Rs)": f"{last * np.exp(r['mu']):,.0f}",
        f"Range {band} (Rs)": f"{last * np.exp(r['lo']):,.0f} - {last * np.exp(r['hi']):,.0f}",
        "Model error": f"{r['mae_model']:.2%}",
        "Naive error": f"{r['mae_naive']:.2%}",
        "Beats naive?": "Yes" if r["mae_model"] < r["mae_naive"] else "No",
        "Direction hit-rate": f"{r['hit']:.0%}",
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("Naive = 'price stays the same'. If the model does not beat it, treat its forecast "
           "as no better than a guess. A direction hit-rate near 50% means no real edge.")

st.warning("Gold is driven by unpredictable events (wars, central banks, inflation shocks). "
           "This is an educational tool, not financial advice.")
