import requests, pandas as pd, numpy as np, json
from datetime import datetime, timezone

# --- Parametría
SYMBOL = "BTCUSDT"

def fetch_klines(symbol, interval, limit=1500):
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
    r.raise_for_status()
    k = r.json()
    df = pd.DataFrame(k, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df[["timestamp","open","high","low","close"]].set_index("timestamp")

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def chandelier_exit_long(close, atr_mult=3, atr_period=22):
    tr = close.diff().abs()
    atr = tr.rolling(atr_period).mean()
    return close.rolling(atr_period).max() - atr_mult * atr

def make_dirs_1d_4h(df1d, df4h):
    # SSL por EMA9/21
    ssl1d = np.where(ema(df1d["close"],9) > ema(df1d["close"],21), 1, -1)
    ssl4h = np.where(ema(df4h["close"],9) > ema(df4h["close"],21), 1, -1)
    # “supertrend” de respaldo: close > EMA10
    st1d = np.where(df1d["close"] > ema(df1d["close"],10), 1, -1)
    st4h = np.where(df4h["close"] > ema(df4h["close"],10), 1, -1)
    # Chandelier bias 1D
    ch_long_exit = chandelier_exit_long(df1d["close"],3,22)
    ch1d = np.where(df1d["close"] > ch_long_exit, 1, -1)

    votes1d = (st1d + ssl1d + ch1d).astype(int)
    dir1d   = np.where(votes1d >= 1, 1, -1)
    votes4h = (st4h + ssl4h).astype(int)
    dir4h   = np.where(votes4h >= 1, 1, -1)

    # Alinear 4H al último voto del día
    v4h_d = pd.Series(votes4h, index=df4h.index).resample("1D").last().reindex(df1d.index).ffill()
    d4h_d = np.where(v4h_d >= 1, 1, -1)

    return votes1d, dir1d, v4h_d.values.astype(int), d4h_d.astype(int)

def volatility_regime(close_1d):
    ret = close_1d.pct_change()
    vol = ret.rolling(20).std().iloc[-1]
    if pd.isna(vol): return "SIDEWAYS"
    if vol > 0.03: return "HIGH_VOL"
    if vol < 0.01: return "LOW_VOL"
    return "SIDEWAYS"

def main():
    # Datos recientes
    df1d = fetch_klines(SYMBOL, "1d")      # ~500 días
    df4h = fetch_klines(SYMBOL, "4h")      # ~500*6 velas 4h

    v1d, d1d, v4d, d4d = make_dirs_1d_4h(df1d, df4h)
    agree = d1d[-1] == d4d[-1]

    # Score muy simple (magnitud de votos + pendiente EMA200)
    ema200 = ema(df1d["close"], 200)
    slope  = ((ema200 - ema200.shift(1)) / ema200).fillna(0.0).clip(-0.02,0.02)
    norm_votes = (0.6*(v1d[-1]/3.0) + 0.4*(v4d[-1]/2.0))
    raw_score = abs(0.85*norm_votes + 0.15*slope.iloc[-1])
    score = raw_score if agree else 0.0

    regime = volatility_regime(df1d["close"])
    direction = "FLAT"
    confidence = 0.50
    if agree:
        direction = "LONG" if d1d[-1] == 1 else "SHORT"
        # política v1 por régimen (suave)
        if regime == "HIGH_VOL" and raw_score < 0.25:
            direction, confidence = "FLAT", 0.50
        else:
            confidence = round(float(min(0.9, 0.55 + 0.35*raw_score)), 2)

    signal = {
        "version": "0.3.0",
        "asset": "BTCUSDT",
        "signal": {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeframe": "1D",
            "horizon_hours": 24,
            "direction": direction,
            "confidence": confidence,
            "risk_score": 0.40 if regime != "HIGH_VOL" else 0.65,
            "volatility_regime": regime,
            "explanation": f"1D votes={int(v1d[-1])}, 4H(d) votes={int(v4d[-1])}, agree={agree}, score={raw_score:.2f}."
        },
        "provenance": {
            "models": ["rules: SSL(ema9/21)+STbackup(ema10)+Chandelier"],
            "data_sources": ["Binance public klines (1D,4H)"]
        }
    }

    with open("btc_analyst_signal_v03.json","w") as f:
        json.dump(signal, f, indent=2)

if __name__ == "__main__":
    main()
