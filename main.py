from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================================
# App
# ==========================================================
app = FastAPI(title="CryptoVision Backend")

# ==========================================================
# CORS
# - Pentru GitHub Pages: permite strict domeniul tău.
# - IMPORTANT: allow_credentials=False => nu e nevoie de wildcard + credentials.
# ==========================================================
ALLOWED_ORIGINS = [
    "https://cryptovisionpaul.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# ==========================================================
# HTTP session + retries (stabilitate + pool)
# ==========================================================
session = requests.Session()

retry = Retry(
    total=2,
    backoff_factor=0.7,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)

adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update(
    {
        "Accept": "application/json",
        "User-Agent": "CryptoVision/1.0 (Render; GitHub Pages frontend)",
    }
)

# ==========================================================
# In-memory cache + throttle (anti-429)
# Render free: cache se poate reseta la restart / redeploy.
# ==========================================================
CACHE_TTL_SECONDS = 60          # markets/predict "fresh" 60 sec
CACHE_STALE_SECONDS = 15 * 60   # stale up to 15 min
THROTTLE_SECONDS = 45           # min interval upstream hits per throttle key

# Pentru CHART folosim TTL separat, mai lung (reduce rate-limit)
CHART_TTL_SECONDS = 5 * 60      # 5 min fresh
CHART_STALE_SECONDS = 60 * 60   # 60 min stale fallback
CHART_THROTTLE_SECONDS = 25     # chart poate fi apelat des din UI; îl protejăm mai agresiv

_cache: Dict[str, Dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _make_cache_key(path: str, params: Dict[str, Any]) -> str:
    # cheie deterministă
    return f"{path}:{json.dumps(params, sort_keys=True, separators=(',', ':'))}"


def _cache_get_fresh(key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    obj = _cache.get(key)
    if not obj:
        return None
    age = _now() - float(obj.get("ts", 0.0))
    if age <= ttl_seconds:
        return obj
    return None


def _cache_get_stale(key: str, stale_seconds: int) -> Optional[Dict[str, Any]]:
    obj = _cache.get(key)
    if not obj:
        return None
    age = _now() - float(obj.get("ts", 0.0))
    if age <= stale_seconds:
        return obj
    return None


def _cache_set(key: str, data: Any, status: int = 200) -> None:
    prev = _cache.get(key, {})
    _cache[key] = {
        "ts": _now(),
        "data": data,
        "status": status,
        "last_upstream_ts": float(prev.get("last_upstream_ts", 0.0)),
    }


def _can_hit_upstream(throttle_key: str, throttle_seconds: int) -> bool:
    obj = _cache.get(throttle_key)
    if not obj:
        return True
    last = float(obj.get("last_upstream_ts", 0.0) or 0.0)
    return (_now() - last) >= throttle_seconds


def _mark_upstream_hit(throttle_key: str) -> None:
    obj = _cache.get(throttle_key)
    if not obj:
        _cache[throttle_key] = {"ts": 0.0, "data": None, "status": 0, "last_upstream_ts": _now()}
    else:
        obj["last_upstream_ts"] = _now()


def _safe_json(resp: requests.Response) -> Tuple[bool, Any]:
    try:
        return True, resp.json()
    except Exception:
        return False, resp.text


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ==========================================================
# Routes
# ==========================================================
@app.get("/")
def root():
    return {"status": "CryptoVision backend is running"}


@app.get("/health")
def health():
    return {"ok": True}


# ==========================================================
# Proxy: /coins/markets
# ==========================================================
@app.get("/coins/markets")
def coins_markets(
    response: Response,
    vs_currency: str = "usd",
    order: str = "market_cap_desc",
    per_page: int = Query(50, ge=1, le=250),
    page: int = Query(1, ge=1, le=50),
    sparkline: bool = False,
    price_change_percentage: str = "1h,24h,7d",
):
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": order,
        "per_page": per_page,
        "page": page,
        "sparkline": str(sparkline).lower(),
        "price_change_percentage": price_change_percentage,
    }

    cache_key = _make_cache_key("/coins/markets", params)
    last_good_key = "LAST_GOOD:/coins/markets"
    throttle_key = "THROTTLE:/coins/markets"

    cached = _cache_get_fresh(cache_key, ttl_seconds=CACHE_TTL_SECONDS)
    if cached:
        response.headers["X-Cache"] = "HIT"
        response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"
        return cached["data"]

    # throttle => serve stale
    if not _can_hit_upstream(throttle_key, throttle_seconds=THROTTLE_SECONDS):
        stale_exact = _cache_get_stale(cache_key, stale_seconds=CACHE_STALE_SECONDS)
        if stale_exact:
            response.headers["X-Cache"] = "STALE-THROTTLE"
            return stale_exact["data"]

        stale_last = _cache_get_stale(last_good_key, stale_seconds=CACHE_STALE_SECONDS)
        if stale_last:
            response.headers["X-Cache"] = "STALE-LAST-GOOD-THROTTLE"
            return stale_last["data"]

        return JSONResponse(
            status_code=429,
            content={"error": "Throttled to protect upstream. No cached data yet."},
        )

    _mark_upstream_hit(throttle_key)

    try:
        r = session.get(url, params=params, timeout=20)

        if r.status_code == 200:
            ok, data = _safe_json(r)
            if not ok or not isinstance(data, list):
                return JSONResponse(status_code=502, content={"error": "Invalid JSON from upstream (CoinGecko)."})
            _cache_set(cache_key, data, status=200)
            _cache_set(last_good_key, data, status=200)

            response.headers["X-Cache"] = "MISS"
            response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"
            return data

        if r.status_code == 429:
            stale_exact = _cache_get_stale(cache_key, stale_seconds=CACHE_STALE_SECONDS)
            if stale_exact:
                response.headers["X-Cache"] = "STALE-EXACT"
                response.headers["X-Upstream-Status"] = "429"
                ra = r.headers.get("Retry-After")
                if ra:
                    response.headers["Retry-After"] = ra
                return stale_exact["data"]

            stale_last = _cache_get_stale(last_good_key, stale_seconds=CACHE_STALE_SECONDS)
            if stale_last:
                response.headers["X-Cache"] = "STALE-LAST-GOOD"
                response.headers["X-Upstream-Status"] = "429"
                ra = r.headers.get("Retry-After")
                if ra:
                    response.headers["Retry-After"] = ra
                return stale_last["data"]

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Upstream rate limited (CoinGecko). No cached data yet.",
                    "detail": (r.text or "")[:500],
                },
            )

        return JSONResponse(
            status_code=502,
            content={
                "error": "Upstream request failed (CoinGecko).",
                "status_code": r.status_code,
                "detail": (r.text or "")[:500],
            },
        )

    except requests.RequestException as e:
        stale_last = _cache_get_stale(last_good_key, stale_seconds=CACHE_STALE_SECONDS)
        if stale_last:
            response.headers["X-Cache"] = "STALE-LAST-GOOD-EXCEPTION"
            response.headers["X-Upstream-Status"] = "EXCEPTION"
            return stale_last["data"]

        return JSONResponse(status_code=503, content={"error": "Upstream request exception (CoinGecko).", "detail": str(e)})


# ==========================================================
# Proxy: /chart
# UI: /chart?coin_id=bitcoin&vs_currency=usd&days=7&interval=hourly
# IMPORTANT: UI chart should use backend to reduce 429 risk.
# ==========================================================
@app.get("/chart")
def chart(
    response: Response,
    coin_id: str = Query(..., min_length=1, max_length=128, description="CoinGecko coin id, ex: bitcoin"),
    vs_currency: str = Query("usd", min_length=1, max_length=16),
    days: int = Query(7, ge=1, le=90),
    interval: str = Query("hourly", description="hourly | daily"),
):
    interval_norm = interval.strip().lower()
    if interval_norm not in ("hourly", "daily"):
        return JSONResponse(status_code=422, content={"error": "Invalid interval. Use 'hourly' or 'daily'."})

    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": interval_norm}

    cache_key = _make_cache_key("/chart", {"coin_id": coin_id, **params})
    throttle_key = f"THROTTLE:/chart:{coin_id}:{days}:{interval_norm}"

    cached = _cache_get_fresh(cache_key, ttl_seconds=CHART_TTL_SECONDS)
    if cached:
        response.headers["X-Cache"] = "HIT"
        response.headers["Cache-Control"] = f"public, max-age={CHART_TTL_SECONDS}"
        return cached["data"]

    if not _can_hit_upstream(throttle_key, throttle_seconds=CHART_THROTTLE_SECONDS):
        stale = _cache_get_stale(cache_key, stale_seconds=CHART_STALE_SECONDS)
        if stale:
            response.headers["X-Cache"] = "STALE-THROTTLE"
            return stale["data"]
        return JSONResponse(status_code=429, content={"error": "Throttled chart request. No cached chart yet."})

    _mark_upstream_hit(throttle_key)

    try:
        r = session.get(url, params=params, timeout=25)

        if r.status_code == 200:
            ok, data = _safe_json(r)
            if not ok or not isinstance(data, dict) or "prices" not in data:
                return JSONResponse(status_code=502, content={"error": "Invalid chart JSON from upstream."})

            # Reduce payload size slightly (optional): keep only prices
            payload = {"prices": data.get("prices") or []}

            _cache_set(cache_key, payload, status=200)

            response.headers["X-Cache"] = "MISS"
            response.headers["Cache-Control"] = f"public, max-age={CHART_TTL_SECONDS}"
            return payload

        if r.status_code == 429:
            stale = _cache_get_stale(cache_key, stale_seconds=CHART_STALE_SECONDS)
            if stale:
                response.headers["X-Cache"] = "STALE"
                response.headers["X-Upstream-Status"] = "429"
                ra = r.headers.get("Retry-After")
                if ra:
                    response.headers["Retry-After"] = ra
                return stale["data"]

            return JSONResponse(status_code=429, content={"error": "Upstream rate limited for chart. No cache yet."})

        return JSONResponse(
            status_code=502,
            content={"error": "Upstream chart failed.", "status_code": r.status_code, "detail": (r.text or "")[:500]},
        )

    except requests.RequestException as e:
        stale = _cache_get_stale(cache_key, stale_seconds=CHART_STALE_SECONDS)
        if stale:
            response.headers["X-Cache"] = "STALE-EXCEPTION"
            response.headers["X-Upstream-Status"] = "EXCEPTION"
            return stale["data"]

        return JSONResponse(status_code=503, content={"error": "Chart upstream exception.", "detail": str(e)})


# ==========================================================
# Indicators + statistical model (non-demo): /ai/predict
# ==========================================================
def _rsi_14(series: List[float]) -> Optional[float]:
    if len(series) < 15:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(-14, 0):
        diff = series[i] - series[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / 14.0
    avg_loss = sum(losses) / 14.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _stdev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(0.0, var))


def _timeframe_to_days(timeframe: str, days: int) -> Tuple[int, str]:
    """
    CoinGecko market_chart supports interval=hourly/daily.
    For small TF: use hourly and clamp days.
    """
    tf = timeframe.strip().lower()
    if tf in ("30m", "30min", "30-min"):
        return _clamp_int(days, 2, 30), "hourly"
    if tf in ("1h", "60m", "1hour"):
        return _clamp_int(days, 2, 60), "hourly"
    if tf in ("4h", "240m", "4hour"):
        return _clamp_int(days, 3, 90), "hourly"
    return _clamp_int(days, 7, 365), "daily"


def _forecast_next_price(closes: List[float], horizon_steps: int = 1) -> Tuple[float, float, float]:
    """
    Model on log-returns:
      mu = mean(log-return), sigma = stdev(log-return)
    Forecast: P_next = P_last * exp(mu*h)
    Interval (68%): exp(±1*sigma*sqrt(h))
    """
    if not closes:
        return 0.0, 0.0, 0.0
    if len(closes) < 3:
        last = float(closes[-1])
        return last, last, last

    rets: List[float] = []
    for i in range(1, len(closes)):
        p0 = float(closes[i - 1])
        p1 = float(closes[i])
        if p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))

    if len(rets) < 2:
        last = float(closes[-1])
        return last, last, last

    mu = _mean(rets)
    sigma = _stdev(rets)
    h = max(1, int(horizon_steps))
    last = float(closes[-1])

    pred = last * math.exp(mu * h)

    spread = sigma * math.sqrt(h)
    lo = last * math.exp((mu - spread) * h)
    hi = last * math.exp((mu + spread) * h)

    return float(pred), float(lo), float(hi)


@app.get("/ai/predict")
def ai_predict(
    response: Response,
    coin_id: str = Query(..., min_length=1, max_length=128, description="CoinGecko coin id, ex: bitcoin"),
    days: int = Query(90, ge=7, le=365, description="History window"),
    timeframe: str = Query("1d", description="30m | 1h | 4h | 1d"),
):
    tf_norm = timeframe.strip().lower()
    days_eff, interval = _timeframe_to_days(tf_norm, days)

    params = {"coin_id": coin_id, "days": days_eff, "interval": interval, "tf": tf_norm}
    cache_key = _make_cache_key("/ai/predict", params)
    throttle_key = f"THROTTLE:/ai/predict:{coin_id}:{tf_norm}"

    cached = _cache_get_fresh(cache_key, ttl_seconds=CACHE_TTL_SECONDS)
    if cached:
        response.headers["X-Cache"] = "HIT"
        response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"
        return cached["data"]

    if not _can_hit_upstream(throttle_key, throttle_seconds=THROTTLE_SECONDS):
        stale = _cache_get_stale(cache_key, stale_seconds=CACHE_STALE_SECONDS)
        if stale:
            response.headers["X-Cache"] = "STALE-THROTTLE"
            return stale["data"]
        return JSONResponse(status_code=429, content={"error": "Throttled AI endpoint. Please wait and retry."})

    _mark_upstream_hit(throttle_key)

    try:
        # 1) current price
        url_price = f"{COINGECKO_BASE}/simple/price"
        r1 = session.get(url_price, params={"ids": coin_id, "vs_currencies": "usd"}, timeout=20)
        if r1.status_code != 200:
            return JSONResponse(
                status_code=502,
                content={"error": "Failed to fetch price", "status": r1.status_code, "detail": (r1.text or "")[:300]},
            )

        ok1, price_json = _safe_json(r1)
        if not ok1 or not isinstance(price_json, dict):
            return JSONResponse(status_code=502, content={"error": "Invalid price JSON from upstream."})

        last_price = (price_json.get(coin_id) or {}).get("usd")
        if last_price is None:
            return JSONResponse(status_code=404, content={"error": "Unknown coin_id or missing price", "coin_id": coin_id})

        # 2) history
        url_chart = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
        r2 = session.get(url_chart, params={"vs_currency": "usd", "days": days_eff, "interval": interval}, timeout=25)

        if r2.status_code == 429:
            stale = _cache_get_stale(cache_key, stale_seconds=CACHE_STALE_SECONDS)
            if stale:
                response.headers["X-Cache"] = "STALE-429"
                response.headers["X-Upstream-Status"] = "429"
                return stale["data"]
            return JSONResponse(status_code=429, content={"error": "Upstream rate-limited (CoinGecko) for history. Try later."})

        if r2.status_code != 200:
            return JSONResponse(
                status_code=502,
                content={"error": "Failed to fetch history", "status": r2.status_code, "detail": (r2.text or "")[:300]},
            )

        ok2, chart_json = _safe_json(r2)
        if not ok2 or not isinstance(chart_json, dict):
            return JSONResponse(status_code=502, content={"error": "Invalid history JSON from upstream."})

        prices = chart_json.get("prices") or []
        closes = [float(p[1]) for p in prices if isinstance(p, list) and len(p) >= 2 and p[1] is not None]

        if len(closes) < 10:
            return JSONResponse(status_code=502, content={"error": "Not enough history data to predict."})

        rsi = _rsi_14(closes)
        predicted, lo68, hi68 = _forecast_next_price(closes, horizon_steps=1)

        # soft RSI adjustment (±0.35%)
        adj = 0.0
        if rsi is not None:
            if rsi < 35:
                adj = +0.0035
            elif rsi > 65:
                adj = -0.0035

        predicted_adj = predicted * (1.0 + adj)

        payload = {
            "coin_id": coin_id,
            "timeframe": tf_norm,
            "days_used": days_eff,
            "interval": interval,
            "last_price": float(last_price),
            "predicted_price": float(predicted_adj),
            "prediction_interval_68": {"low": float(lo68), "high": float(hi68)},
            "indicators": {"rsi": rsi},
            "model": "statistical-returns-v1",
            "note": "Statistical forecast (log-returns + volatility). Educational only.",
        }

        _cache_set(cache_key, payload, status=200)
        response.headers["X-Cache"] = "MISS"
        response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"
        return payload

    except requests.RequestException as e:
        stale = _cache_get_stale(cache_key, stale_seconds=CACHE_STALE_SECONDS)
        if stale:
            response.headers["X-Cache"] = "STALE-EXCEPTION"
            response.headers["X-Upstream-Status"] = "EXCEPTION"
            return stale["data"]

        return JSONResponse(status_code=503, content={"error": "AI upstream exception.", "detail": str(e)})


# ==========================================================
# Backward compatibility: demo endpoint (rule-based)
# ==========================================================
@app.get("/demo/prediction")
def demo_prediction(
    coin_id: str = Query(..., description="CoinGecko coin id, ex: bitcoin"),
    days: int = Query(90, ge=7, le=365, description="History window"),
):
    url_price = f"{COINGECKO_BASE}/simple/price"
    r1 = session.get(url_price, params={"ids": coin_id, "vs_currencies": "usd"}, timeout=20)
    if r1.status_code != 200:
        return {"error": "Failed to fetch price", "coin_id": coin_id, "status": r1.status_code, "body": r1.text}

    ok, price_json = _safe_json(r1)
    if not ok or not isinstance(price_json, dict):
        return {"error": "Invalid JSON from upstream", "coin_id": coin_id}

    last_price = (price_json.get(coin_id) or {}).get("usd")
    if last_price is None:
        return {"error": "Unknown coin_id or missing price", "coin_id": coin_id, "raw": price_json}

    url_chart = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    r2 = session.get(url_chart, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=25)
    if r2.status_code != 200:
        return {"error": "Failed to fetch history", "coin_id": coin_id, "status": r2.status_code, "body": r2.text}

    ok2, chart_json = _safe_json(r2)
    if not ok2 or not isinstance(chart_json, dict):
        return {"error": "Invalid history JSON", "coin_id": coin_id}

    prices = (chart_json.get("prices") or [])
    closes = [p[1] for p in prices if isinstance(p, list) and len(p) >= 2]
    rsi = _rsi_14([float(x) for x in closes if x is not None])

    drift = 0.003
    label = "demo-neutral"
    if rsi is not None:
        if rsi < 40:
            drift = 0.015
            label = "demo-rebound"
        elif rsi > 60:
            drift = -0.010
            label = "demo-cooldown"

    predicted_price = float(last_price) * (1.0 + drift)

    return {
        "coin_id": coin_id,
        "model": label,
        "days": days,
        "last_price": float(last_price),
        "predicted_price": float(predicted_price),
        "indicators": {"rsi": rsi},
        "note": "Demo endpoint (rule-based). Prefer /ai/predict.",
    }
