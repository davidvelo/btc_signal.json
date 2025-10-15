import requests, pandas as pd, numpy as np, json
from datetime import datetime, timezone

SYMBOL = "BTCUSDT"  # lo mantenemos para nombrado; los proveedores pueden usar USD

# ---------- Helpers ----------
def to_df(rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close"]]

def fetch_bitstamp(limit=1000, step_sec=3600, pair="btcusdt"):
    # step=3600 => 1H; luego re-sampleamos a 4H y 1D
    url = "https://www.bitstamp.net/api/v2/ohlc/{}/".format(pair)
    r = requests.get(url, params={"step": step_sec, "limit": limit}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "data" not in data or "ohlc" not in data["data"]:
        raise ValueError("Bitstamp bad payload")
    rows = []
    for o in data["data"]["ohlc"]:
        ts = datetime.fromtimestamp(int(o["timestamp"]), tz=timezone.utc).isoformat()
        rows.append([ts, o["open"], o["high"], o["low"], o["close"]])
    return to_df(rows, ["timestamp","open","high","low","close"])

def fetch_coinbase(granularity=3600, limit=300, product="BTC-USD"):
    # Coinbase devuelve [time, low, high, open, close, volume] en orden inverso
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    r = requests.get(url, params={"granularity": granularity, "limit": limit}, timeout=20,
                     headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    arr = r.json()
    if not isinstance(arr, list):
        raise ValueError("Coinbase bad payload")
    rows = []
    for t, low, high, open_, close, _ in arr:
        ts = datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()
        rows.append([ts, open_, high, low, close])
    return to_df(rows, ["timestamp","open","high","low","close"])

def fetch_kraken(pair="XXBTZUSD", interval=60, limit=1000):
    url = "https://api.kraken.com/0/public/OHLC"
    r = requests.get(url, params={"pair": pair, "interval": interval}, timeout=20)
    r.raise_for_status()
    data = r.json()
    key = list(data["result"].keys())[0]
    arr = data["result"][key]
    rows = []
    for t, o, h, l, c, *_ in arr[-limit:]:
        ts = datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()
        rows.append([ts, o, h, l, c])
    return to_df(rows, ["timestamp","open","high","low","close"])

def fetch_ohlc_fallback():
    # Intentamos Bitstamp (USDT), si falla Coinbase (USD), si falla Kraken (USD)
    errors = []
    for fn in [
        lambda: fetch_bitstamp(pair="btcusdt"),
        lambda: fetch_coinbase(product="BTC-USD"),
        lambda: fetch_kraken(pair="XXBTZUSD")
    ]:
        try:
            return fn()
        except Exception as e:
            errors.append(str(e))
            continue
    raise RuntimeError("All providers failed: " + " | ".join(errors))

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def chandelier_exit_long(close, atr_mult=3, atr_period=22):
    tr = close.diff().abs()
    atr = tr.rolling(atr_period).mean()
    return close.rolling(atr_period).max() - atr_mult * atr

def make_dirs_1d_4h(df):
    # Tenemos 1H (o variable) → construimos vistas 4H y 1D
    df4h = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    df1d = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()

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

    # Alinear 4H a 1D (último voto del día)
    v4h_d = pd.Series(votes4h, index=df4h.index).resample("1D").last().reindex(df1d.index).ffill()
    d4h_d = np.where(v4h_d >= 1, 1, -1)

    return df1d, df4h, votes1d, dir1d, v4h_d.values.astype(int), d4h_d.astype(int)

def volatility_regime(close_1d):
    ret = close_1d.pct_change()
    vol = ret.rolling(20).std().iloc[-1]
    if pd.isna(vol): return "SIDEWAYS"
    if vol > 0.03: return "HIGH_VOL"
    if vol < 0.01: return "LOW_VOL"
    return "SIDEWAYS"

def main():
    df = fetch_ohlc_fallback()              # 1H (Bitstamp/CB/Kraken)
    df1d, df4h, v1d, d1d, v4d, d4d = make_dirs_1d_4h(df)
    agree = d1d[-1] == d4d[-1]

    # Score simple
    ema200 = ema(df1d["close"], 200)
    slope  = ((ema200 - ema200.shift(1)) / ema200).fillna(0.0).clip(-0.02,0.02)
    norm_votes = (0.6*(v1d[-1]/3.0) + 0.4*(v4d[-1]/2.0))
    raw_score = abs(0.85*norm_votes + 0.15*slope.iloc[-1])
    score = raw_score if agree else 0.0

    regime = volatility_regime(df1d["close"])
    direction, confidence = "FLAT", 0.50
    if agree:
        direction = "LONG" if d1d[-1] == 1 else "SHORT"
        if regime == "HIGH_VOL" and raw_score < 0.25:
            direction, confidence = "FLAT", 0.50
        else:
            confidence = round(float(min(0.9, 0.55 + 0.35*raw_score)), 2)

    signal = {
        "version": "0.3.0",
        "asset": SYMBOL,
        "signal": {
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeframe": "1D",
            "horizon_hours": 24,
            "direction": direction,
            "confidence": confidence,
            "risk_score": 0.40 if regime != "HIGH_VOL" else 0.65,
            "volatility_regime": regime,
            "explanation": f"1D votes={int(v1d[-1])}, 4H(d) votes={int(v4d[-1])}, agree={agree}, score={raw_score:.2f} (fuente OHLC fallback)."
        },
        "provenance": {
            "models": ["rules: SSL(ema9/21)+STbackup(ema10)+Chandelier"],
            "data_sources": ["Bitstamp/CB/Kraken OHLC (fallback)"]
        }
    }

    with open("btc_analyst_signal_v03.json","w") as f:
        json.dump(signal, f, indent=2)

if __name__ == "__main__":
    main()
