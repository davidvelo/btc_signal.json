# backtest.py
import json, requests, numpy as np, pandas as pd
from datetime import datetime, timezone

# ------------------ Data sources (fallback) ------------------
def _to_df(rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close"]]

def fetch_bitstamp(limit=1000, step_sec=3600, pair="btcusdt"):
    url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/"
    r = requests.get(url, params={"step": step_sec, "limit": limit}, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    for o in data["data"]["ohlc"]:
        ts = datetime.fromtimestamp(int(o["timestamp"]), tz=timezone.utc).isoformat()
        rows.append([ts, o["open"], o["high"], o["low"], o["close"]])
    return _to_df(rows, ["timestamp","open","high","low","close"])

def fetch_coinbase(granularity=3600, limit=300, product="BTC-USD"):
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    r = requests.get(url, params={"granularity": granularity, "limit": limit}, timeout=20,
                     headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    arr = r.json()
    rows = []
    for t, low, high, open_, close_, _ in arr:
        ts = datetime.fromtimestamp(int(t), tz=timezone.utc).isoformat()
        rows.append([ts, open_, high, low, close_])
    return _to_df(rows, ["timestamp","open","high","low","close"])

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
    return _to_df(rows, ["timestamp","open","high","low","close"])

def fetch_ohlc_fallback():
    for fn in [
        lambda: fetch_bitstamp(pair="btcusdt"),
        lambda: fetch_coinbase(product="BTC-USD"),
        lambda: fetch_kraken(pair="XXBTZUSD")
    ]:
        try:
            return fn()
        except Exception:
            continue
    raise RuntimeError("All providers failed.")

# ------------------ Indicators & rules ------------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def chandelier_exit_long(close, atr_mult=3, atr_period=22):
    tr = close.diff().abs()
    atr = tr.rolling(atr_period).mean()
    return close.rolling(atr_period).max() - atr_mult * atr

def make_frames():
    df = fetch_ohlc_fallback()                                # 1H aprox
    df4h = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    df1d = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    return df1d, df4h

def base_consensus(df1d, df4h):
    # SSL
    ssl1d = np.where(ema(df1d["close"],9) > ema(df1d["close"],21), 1, -1)
    ssl4h = np.where(ema(df4h["close"],9) > ema(df4h["close"],21), 1, -1)
    # ST backup
    st1d = np.where(df1d["close"] > ema(df1d["close"],10), 1, -1)
    st4h = np.where(df4h["close"] > ema(df4h["close"],10), 1, -1)
    # Chandelier bias
    ch_exit = chandelier_exit_long(df1d["close"],3,22)
    ch1d = np.where(df1d["close"] > ch_exit, 1, -1)

    votes1d = (st1d + ssl1d + ch1d).astype(int)
    dir1d   = np.where(votes1d >= 1, 1, -1)
    votes4h = (st4h + ssl4h).astype(int)
    dir4h   = np.where(votes4h >= 1, 1, -1)

    # Alinear 4H→1D (último del día)
    v4h_d = pd.Series(votes4h, index=df4h.index).resample("1D").last().reindex(df1d.index).ffill()
    d4h_d = np.where(v4h_d >= 1, 1, -1)

    # Score sencillo (magnitud de votos + pendiente EMA200)
    ema200 = ema(df1d["close"],200)
    slope  = ((ema200 - ema200.shift(1)) / ema200).fillna(0.0).clip(-0.02,0.02)
    norm_votes = (0.6*(votes1d/3.0) + 0.4*(v4h_d/2.0))
    score = (0.85*np.abs(norm_votes) + 0.15*np.abs(slope)).astype(float)

    # Régimen
    vol20 = df1d["close"].pct_change().rolling(20).std()
    regime = pd.Series("SIDEWAYS", index=df1d.index)
    regime[vol20 > 0.03] = "HIGH_VOL"
    regime[vol20 < 0.01] = "LOW_VOL"

    out = pd.DataFrame(index=df1d.index)
    out["close"] = df1d["close"]
    out["votes1d"] = votes1d
    out["dir1d"] = dir1d
    out["votes4h_d"] = v4h_d.astype(int)
    out["dir4h_d"] = d4h_d.astype(int)
    out["agree"] = (out["dir1d"] == out["dir4h_d"]).astype(int)
    out["score"] = score
    out["regime"] = regime
    out["next_ret"] = out["close"].pct_change().shift(-1)
    return out

# ------------------ Walk-forward tuner ------------------
GRID_SCORE = [0.10, 0.15, 0.20, 0.25, 0.30]
GRID_V1D   = [1, 2, 3]
GRID_V4H   = [1, 2]

def apply_policy(df, th):
    smin, v1m, v4m = th["score_min"], th["votes1d_min"], th["votes4h_min"]
    mask = (df["agree"]==1) & (df["score"]>=smin) & (np.abs(df["votes1d"])>=v1m) & (np.abs(df["votes4h_d"])>=v4m)
    sig = np.where(mask, df["dir1d"], 0)
    return pd.Series(sig, index=df.index)

def objective(sig, ret):
    active = sig!=0
    if active.mean()==0: 
        return -1e9, 0.0, 0.0, 0.0
    edge = np.nanmean(sig*ret)
    pnl = (sig.shift(1).fillna(0)*ret).fillna(0)  # ejecución al cierre previo
    sharpe = np.nanmean(pnl)/ (np.nanstd(pnl)+1e-9) * np.sqrt(252)
    return float(edge), float(active.mean()), float(sharpe), float(pnl.cumsum().iloc[-1])

def tune_thresholds(train_df):
    best = {"objective": -1e9}
    for s in GRID_SCORE:
        for v1 in GRID_V1D:
            for v4 in GRID_V4H:
                th = {"score_min": s, "votes1d_min": v1, "votes4h_min": v4}
                sig = apply_policy(train_df, th)
                edge, cov, shp, eq = objective(sig, train_df["next_ret"])
                obj = edge + 0.000 * cov + 0.00 * shp  # simple: prioriza edge
                if obj > best["objective"]:
                    best = {"objective": obj, "score_min": s, "votes1d_min": v1, "votes4h_min": v4,
                            "edge": edge, "coverage": cov, "sharpe": shp, "equity_last": eq}
    return best

def walk_forward(df, train_days=180, test_days=30):
    i = 0
    rows = []
    while True:
        start = df.index[0] + pd.Timedelta(days=i*test_days)
        mid   = start + pd.Timedelta(days=train_days)
        end   = mid + pd.Timedelta(days=test_days)
        train = df.loc[(df.index>=start)&(df.index<mid)]
        test  = df.loc[(df.index>=mid)&(df.index<end)]
        if len(test)<5: break
        th = tune_thresholds(train)
        sig_test = apply_policy(test, th)
        edge, cov, shp, eq = objective(sig_test, test["next_ret"])
        rows.append({
            "train_start": str(train.index[0]) if len(train) else None,
            "train_end":   str(train.index[-1]) if len(train) else None,
            "test_start":  str(test.index[0]),
            "test_end":    str(test.index[-1]),
            "score_min": th["score_min"], "votes1d_min": th["votes1d_min"], "votes4h_min": th["votes4h_min"],
            "edge": edge, "coverage": cov, "sharpe": shp, "equity_last": eq
        })
        i += 1
    return pd.DataFrame(rows)

def main():
    df1d, df4h = make_frames()
    base = base_consensus(df1d, df4h)

    # Walk-forward
    wf = walk_forward(base, 180, 30)

    # Métricas globales (concatenando todas las ventanas de test con su señal recalculada)
    global_sig = pd.Series(0, index=base.index)
    i = 0
    while True:
        start = base.index[0] + pd.Timedelta(days=i*30)
        mid   = start + pd.Timedelta(days=180)
        end   = mid + pd.Timedelta(days=30)
        train = base.loc[(base.index>=start)&(base.index<mid)]
        test  = base.loc[(base.index>=mid)&(base.index<end)]
        if len(test)<5: break
        th = tune_thresholds(train)
        global_sig.loc[test.index] = apply_policy(test, th)
        i += 1

    edge, cov, shp, eq = objective(global_sig, base["next_ret"])
    pnl = (global_sig.shift(1).fillna(0)*base["next_ret"]).fillna(0)
    equity = (1+pnl).cumprod()

    # Salidas
    reports_dir = "reports"
    import os; os.makedirs(reports_dir, exist_ok=True)

    wf_path = f"{reports_dir}/wf_windows.csv"
    wf.to_csv(wf_path, index=False)

    daily_path = f"{reports_dir}/daily_signals.csv"
    pd.DataFrame({
        "close": base["close"],
        "votes1d": base["votes1d"],
        "votes4h_d": base["votes4h_d"],
        "score": base["score"],
        "regime": base["regime"],
        "signal": global_sig,
        "next_ret": base["next_ret"],
        "equity": equity
    }).to_csv(daily_path)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "samples_total": int(base.shape[0]),
        "coverage_pct": round(cov*100,2),
        "edge_pct": round(edge*100,4),
        "sharpe_like": round(shp,3),
        "equity_last": float(equity.iloc[-1]),
        "train_days": 180, "test_days": 30,
        "grid": {"score": GRID_SCORE, "votes1d": GRID_V1D, "votes4h": GRID_V4H},
        "artifacts": {"wf_windows_csv": wf_path, "daily_signals_csv": daily_path}
    }
    with open(f"{reports_dir}/summary.json","w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
