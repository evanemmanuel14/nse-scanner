import os
import json
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

# =========================
# CONFIG
# =========================
TZ = ZoneInfo("Asia/Kolkata")

INTERVAL = "5m"          # 5m is most reliable for yfinance intraday
PERIOD = "5d"            # enough bars for rolling indicators

SCORE_MIN = 7            # alert threshold
VOL_MULT = 1.3           # ✅ your request (volume pickup threshold)

# Cooldown is optional now (since once/day rule is strict),
# but we keep it as extra protection for edge cases.
COOLDOWN_MIN = 45

MKT_OPEN = time(9, 15)
MKT_CLOSE = time(15, 30)

# 20-symbol watchlist (replace with yours anytime)
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS","PFC.NS", "RECLTD.NS"
]

TG_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

STATE_PATH = os.path.join(os.path.dirname(__file__), "state.json")


# =========================
# Time helpers
# =========================
def now_ist() -> datetime:
    return datetime.now(TZ)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def in_market_hours(dt_ist: datetime) -> bool:
    t = dt_ist.time()
    return (t >= MKT_OPEN) and (t <= MKT_CLOSE)

def today_ist_str() -> str:
    return now_ist().strftime("%Y-%m-%d")


# =========================
# Telegram
# =========================
def send_telegram(text: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Telegram not configured (missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code != 200:
            print("Telegram send failed:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram send exception:", e)


# =========================
# Persistent state
# =========================
def load_state() -> dict:
    """
    state.json structure:
    {
      "RELIANCE.NS": {
        "cooldown_until": "2025-12-19T10:00:00Z",
        "alerts": {
          "PULLBACK_LONG_2025-12-19": true,
          "PULLBACK_SHORT_2025-12-19": true
        }
      },
      ...
    }
    """
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)

def get_cooldown_until(state: dict, sym: str):
    v = state.get(sym, {}).get("cooldown_until")
    if not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception:
        return None

def set_cooldown_until(state: dict, sym: str, dt_utc: datetime):
    state.setdefault(sym, {})
    state[sym]["cooldown_until"] = dt_utc.isoformat().replace("+00:00", "Z")

def in_cooldown(state: dict, sym: str) -> bool:
    cu = get_cooldown_until(state, sym)
    return cu is not None and utc_now() < cu

def already_alerted_today(state: dict, sym: str, sig_type: str) -> bool:
    # once per day per symbol per direction
    day = today_ist_str()
    key = f"{sig_type}_{day}"
    return bool(state.get(sym, {}).get("alerts", {}).get(key, False))

def mark_alerted_today(state: dict, sym: str, sig_type: str):
    day = today_ist_str()
    key = f"{sig_type}_{day}"
    state.setdefault(sym, {})
    state[sym].setdefault("alerts", {})
    state[sym]["alerts"][key] = True

def prune_old_alerts(state: dict, keep_days: int = 10):
    """
    Keep state.json small by keeping only last N days of alert keys.
    """
    cutoff = (now_ist().date() - timedelta(days=keep_days))
    for sym, blob in list(state.items()):
        alerts = blob.get("alerts", {})
        if not isinstance(alerts, dict):
            continue
        new_alerts = {}
        for k, v in alerts.items():
            # key format: TYPE_YYYY-MM-DD
            parts = k.rsplit("_", 1)
            if len(parts) != 2:
                continue
            try:
                d = datetime.strptime(parts[1], "%Y-%m-%d").date()
            except Exception:
                continue
            if d >= cutoff:
                new_alerts[k] = v
        blob["alerts"] = new_alerts
        state[sym] = blob


# =========================
# Data + indicators
# =========================
def fetch_intraday(symbols: list[str]) -> dict[str, pd.DataFrame]:
    df = yf.download(
        tickers=" ".join(symbols),
        interval=INTERVAL,
        period=PERIOD,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    out = {}
    if df is None or len(df) == 0:
        return out

    if isinstance(df.columns, pd.MultiIndex):
        for s in symbols:
            if s in df.columns.get_level_values(0):
                d = df[s].dropna()
                if not d.empty:
                    out[s] = d
    else:
        out[symbols[0]] = df.dropna()

    return out

def ensure_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if getattr(idx, "tz", None) is None:
        df = df.tz_localize("UTC")
    return df.tz_convert(TZ)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Typical price
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].astype(float)

    # VWAP (cumulative over fetched window; good enough for confirmation)
    df["VWAP"] = (tp * vol).cumsum() / (vol.cumsum().replace(0, np.nan))

    # VWMA helper
    def vwma(close, v, length):
        num = (close * v).rolling(length).sum()
        den = v.rolling(length).sum()
        return num / den

    df["VWMA20"] = vwma(df["Close"], vol, 20)
    df["VWMA50"] = vwma(df["Close"], vol, 50)

    # ATR(14)
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Volume baseline
    df["VolAvg20"] = vol.rolling(20).mean()

    return df


# =========================
# Signal logic (Pullback + Trend)
# =========================
def is_trending_up(latest) -> bool:
    if any(pd.isna(latest[x]) for x in ["VWAP", "VWMA20", "VWMA50"]):
        return False
    return (latest["Close"] > latest["VWAP"]) and (latest["VWMA20"] > latest["VWMA50"])

def is_trending_down(latest) -> bool:
    if any(pd.isna(latest[x]) for x in ["VWAP", "VWMA20", "VWMA50"]):
        return False
    return (latest["Close"] < latest["VWAP"]) and (latest["VWMA20"] < latest["VWMA50"])

def recent_swing_high(df: pd.DataFrame, lookback=18) -> float:
    sub = df.iloc[-(lookback+1):-1]
    if sub.empty:
        return np.nan
    return float(sub["High"].max())

def recent_swing_low(df: pd.DataFrame, lookback=18) -> float:
    sub = df.iloc[-(lookback+1):-1]
    if sub.empty:
        return np.nan
    return float(sub["Low"].min())

def pullback_long_signal(df: pd.DataFrame):
    if len(df) < 60:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    if not is_trending_up(latest):
        return None

    swing = recent_swing_high(df, lookback=18)
    if math.isnan(swing):
        return None

    # Pullback touch within last 6 candles (~30 mins on 5m)
    recent = df.iloc[-7:-1]
    touched_vwap = (recent["Low"] <= (recent["VWAP"] * 1.002)).any()   # within 0.2%
    touched_vwma = (recent["Low"] <= (recent["VWMA20"] * 1.002)).any()

    # Continuation trigger: close breaks swing high with small buffer
    broke_swing = latest["Close"] > swing * 1.001  # 0.1% buffer

    # Strength candle: green + close rising
    strength = (latest["Close"] > prev["Close"]) and (latest["Close"] > latest["Open"])

    if not ((touched_vwap or touched_vwma) and strength and broke_swing):
        return None

    score, reasons = 0, []
    score += 3; reasons.append("Uptrend (Close>VWAP, VWMA20>VWMA50)")
    score += 2; reasons.append("Pullback touched VWAP/VWMA20")
    score += 2; reasons.append("Continuation break above swing high")

    if pd.notna(latest["VolAvg20"]) and latest["VolAvg20"] > 0 and latest["Volume"] > VOL_MULT * latest["VolAvg20"]:
        score += 1; reasons.append(f"Volume pickup (>{VOL_MULT}x avg)")

    if pd.notna(latest["ATR14"]) and latest["ATR14"] > 0:
        score += 1; reasons.append("ATR ok")

    return {"type": "PULLBACK_LONG", "score": score, "reasons": reasons, "price": float(latest["Close"])}

def pullback_short_signal(df: pd.DataFrame):
    if len(df) < 60:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    if not is_trending_down(latest):
        return None

    swing = recent_swing_low(df, lookback=18)
    if math.isnan(swing):
        return None

    recent = df.iloc[-7:-1]
    touched_vwap = (recent["High"] >= (recent["VWAP"] * 0.998)).any()
    touched_vwma = (recent["High"] >= (recent["VWMA20"] * 0.998)).any()

    broke_swing = latest["Close"] < swing * 0.999  # 0.1% buffer
    strength = (latest["Close"] < prev["Close"]) and (latest["Close"] < latest["Open"])

    if not ((touched_vwap or touched_vwma) and strength and broke_swing):
        return None

    score, reasons = 0, []
    score += 3; reasons.append("Downtrend (Close<VWAP, VWMA20<VWMA50)")
    score += 2; reasons.append("Pullback touched VWAP/VWMA20")
    score += 2; reasons.append("Continuation break below swing low")

    if pd.notna(latest["VolAvg20"]) and latest["VolAvg20"] > 0 and latest["Volume"] > VOL_MULT * latest["VolAvg20"]:
        score += 1; reasons.append(f"Volume pickup (>{VOL_MULT}x avg)")

    if pd.notna(latest["ATR14"]) and latest["ATR14"] > 0:
        score += 1; reasons.append("ATR ok")

    return {"type": "PULLBACK_SHORT", "score": score, "reasons": reasons, "price": float(latest["Close"])}


def format_alert(symbol: str, sig: dict, bar_time_ist: datetime) -> str:
    direction = "🟢 LONG" if sig["type"] == "PULLBACK_LONG" else "🔴 SHORT"
    reasons = "\n".join([f"• {r}" for r in sig["reasons"]])
    return (
        f"{direction} Pullback Trend Alert\n\n"
        f"Symbol: {symbol}\n"
        f"TF: {INTERVAL}\n"
        f"Time (IST): {bar_time_ist.strftime('%Y-%m-%d %H:%M')}\n"
        f"Price: {sig['price']:.2f}\n"
        f"Score: {sig['score']}/10\n\n"
        f"{reasons}\n\n"
        f"Rule: Only 1 alert/day per symbol per direction.\n"
        f"Note: yfinance intraday can be delayed—confirm on chart before entry."
    )


# =========================
# Main
# =========================
def main():
    dt_ist = now_ist()
    # Only run during market hours
    if not in_market_hours(dt_ist):
        print("Outside NSE market hours (IST) — exiting.")
        return

    state = load_state()
    prune_old_alerts(state, keep_days=10)

    frames = fetch_intraday(WATCHLIST)
    if not frames:
        print("No data returned.")
        save_state(state)
        return

    alerts_sent = 0

    for sym in WATCHLIST:
        df = frames.get(sym)
        if df is None or df.empty:
            continue

        df = ensure_ist_index(df).dropna()
        if len(df) < 60:
            continue

        df = compute_indicators(df)
        bar_time_ist = df.index[-1].to_pydatetime()

        sig = pullback_long_signal(df) or pullback_short_signal(df)
        if not sig:
            continue

        if sig["score"] < SCORE_MIN:
            continue

        sig_type = sig["type"]

        # ✅ Once per day per symbol per direction
        if already_alerted_today(state, sym, sig_type):
            continue

        # Extra cooldown (optional)
        if in_cooldown(state, sym):
            continue

        # Dedupe within this single run (safety)
        run_key = f"{sym}:{sig_type}:{bar_time_ist.strftime('%Y%m%d%H%M')}"
        if run_key in _seen_alerts:
            continue
        _seen_alerts.add(run_key)

        # Send alert
        msg = format_alert(sym, sig, bar_time_ist)
        send_telegram(msg)
        alerts_sent += 1

        # Persist state
        mark_alerted_today(state, sym, sig_type)
        set_cooldown_until(state, sym, utc_now() + timedelta(minutes=COOLDOWN_MIN))

    save_state(state)
    print(f"Done. Alerts sent: {alerts_sent}")


if __name__ == "__main__":
    main()
