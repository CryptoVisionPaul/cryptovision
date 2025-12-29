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
# CORS (GitHub Pages origin)
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
# HTTP session + retries
# ==========================================================
session = requests.Session()

retry = Retry(
    total=2,
    backoff_factor=0.7,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    raise_on_status=False,
    respect_retry_after_header=True,
)

adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update(
    {
        "Accept": "application/json",
        "User-Agent": "CryptoVision/1.1 (Render; GitHub Pages frontend)",
    }
)

# ==========================================================
# In-memory cache + throttle (anti-429)
# ==========================================================
CACHE_TTL_SECONDS = 60          # markets/predict fresh 60 sec
CACHE_STALE_SECONDS = 15 * 60   # stale up to 15 min
THROTTLE_SECONDS = 45           # min interval upstream hits per throttle key

CHART_TTL_SECONDS = 5 * 60      # 5 min fresh
CHART_STALE_SECONDS = 60 * 60   # 60 min stale fallback
CHART_THROTTLE_SECONDS = 25     # protect aggressively

_cache: Dict[str, Dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _make_cache_key(path: str, params: Dict[str, Any]) -> str:
    return f"{path}:{json.dumps(params, sort_keys=True, separators=(',', ':'))}"


def _cache_get_fresh(key: str, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    obj = _cache.get(key)
    if not obj:
        return None
    age = _now() - float(obj.get("ts", 0.0))
    return obj if age <= ttl_seconds else None


def _cache_get_stale(key: str, stale_seconds: int) -> Optional[Dict[str, Any]]:
    obj = _cache.get(key)
    if not obj:
        return None
    age = _now() - float(obj.get("ts", 0.0))
    return obj if age <= stale_seconds else None


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

    if not _can_hit_upstream(throttle_key, throttle_seconds=THROTTLE_SECONDS):
        stale_exact = _cache_get_stale(cache_key, stale_seconds=CACHE_STALE_SECONDS)
        if stale_exact:
            response.headers["X-Cache"] = "STALE-THROTTLE"
            return stale_exact["data"]

        stale_last = _cache_get_stale(last_good_key, stale_seconds=CACHE_STALE_SECONDS)
        if stale_last:
            response.headers["X-Cache"] = "STALE-LAST-GOOD-THROTTLE"
            return stale_last["data"]

        response.headers["Retry-After"] = "15"
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

            ra = r.headers.get("Retry-After") or "20"
            response.headers["Retry-After"] = ra
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
# Always coerce interval to daily for reliability
# ==========================================================
@app.get("/chart")
def chart(
    response: Response,
    coin_id: str = Query(..., min_length=1, max_length=128, description="CoinGecko coin id, ex: bitcoin"),
    vs_currency: str = Query("usd", min_length=1, max_length=16),
    days: int = Query(30, ge=1, le=365),
    interval: str = Query("daily", description="hourly | daily (hourly will be coerced to daily)"),
):
    interval_norm = "daily"
    if interval.strip().lower() == "hourly":
        response.headers["X-Interval-Coerced"] = "hourly->daily"

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
        response.headers["Retry-After"] = "15"
        return JSONResponse(status_code=429, content={"error": "Throttled chart request. No cached chart yet."})

    _mark_upstream_hit(throttle_key)

    try:
        r = session.get(url, params=params, timeout=25)

        if r.status_code == 200:
            ok, data = _safe_json(r)
            if not ok or not isinstance(data, dict) or "prices" not in data:
                return JSONResponse(status_code=502, content={"error": "Invalid chart JSON from upstream."})

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

            response.headers["Retry-After"] = r.headers.get("Retry-After") or "30"
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
# Indicators + statistical model: /ai/predict
# - Uses DAILY market_chart always
# - Forecast horizon expressed in days (fraction allowed)
# - Includes volatility floor so intervals are not unrealistically tight
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


def _timeframe_params(timeframe: str, days: int) -> Tuple[int, str, float]:
    tf = timeframe.strip().lower()
    interval = "daily"
    if tf in ("30m", "30min", "30-min"):
        return _clamp_int(days, 30, 365), interval, 0.5 / 24.0
    if tf in ("1h", "60m", "1hour"):
        return _clamp_int(days, 30, 365), interval, 1.0 / 24.0
    if tf in ("4h", "240m", "4hour"):
        return _clamp_int(days, 30, 365), interval, 4.0 / 24.0
    return _clamp_int(days, 30, 365), interval, 1.0


def _sigma_floor_for_horizon(h_days: float, min_move_rel: float = 0.005) -> float:
    """
    Ensures that for the given horizon h (in days), the 1-sigma band is not absurdly tight.
    For log-returns, a move of 'min_move_rel' corresponds roughly to log(1+min_move_rel).

    Want: sigma * sqrt(h) >= log(1+min_move_rel)
    => sigma >= log(1+min_move_rel) / sqrt(h)
    """
    h = max(1e-6, float(h_days))
    target = math.log(1.0 + float(min_move_rel))
    return target / math.sqrt(h)


def _forecast_next_price(
    closes: List[float],
    h_days: float = 1.0,
    rsi: Optional[float] = None,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Model on DAILY log-returns.

    Correct formulas:
      pred = last * exp(mu*h)
      lo/hi (68%) = last * exp(mu*h ± sigma*sqrt(h))

    Adds:
      - volatility floor so intervals aren't unrealistically tight
      - RSI adjustment applied as a small drift tweak (log-space)

    Returns:
      pred, lo68, hi68, diagnostics
    """
    if not closes:
        return 0.0, 0.0, 0.0, {"mu": 0.0, "sigma": 0.0, "sigma_floor": 0.0}

    last = float(closes[-1])
    if len(closes) < 3 or last <= 0:
        return last, last, last, {"mu": 0.0, "sigma": 0.0, "sigma_floor": 0.0}

    rets: List[float] = []
    for i in range(1, len(closes)):
        p0 = float(closes[i - 1])
        p1 = float(closes[i])
        if p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))

    if len(rets) < 2:
        return last, last, last, {"mu": 0.0, "sigma": 0.0, "sigma_floor": 0.0}

    mu = _mean(rets)
    sigma = _stdev(rets)

    h = max(1e-6, float(h_days))

    # RSI drift adjustment (small, bounded)
    # Interpret as small relative move; convert to log drift approximately.
    # +/- 0.35% as in your code, but applied consistently in log space.
    rsi_adj_rel = 0.0
    if rsi is not None:
        if rsi < 35:
            rsi_adj_rel = +0.0035
        elif rsi > 65:
            rsi_adj_rel = -0.0035

    # Convert to log drift. For small x, log(1+x) ≈ x.
    mu_adj = mu + math.log(1.0 + rsi_adj_rel) / max(h, 1e-6)

    # Volatility floor to avoid "from == to" situations in UI
    sigma_floor = _sigma_floor_for_horizon(h, min_move_rel=0.005)
    sigma_eff = max(sigma, sigma_floor)

    # Predicted (center)
    pred = last * math.exp(mu_adj * h)

    # 68% band around center with correct sqrt(h)
    band = sigma_eff * math.sqrt(h)
    lo = last * math.exp(mu_adj * h - band)
    hi = last * math.exp(mu_adj * h + band)

    # Sanity
    lo = min(lo, hi)
    hi = max(lo, hi)

    return float(pred), float(lo), float(hi), {
        "mu": float(mu),
        "mu_adj": float(mu_adj),
        "sigma": float(sigma),
        "sigma_eff": float(sigma_eff),
        "sigma_floor": float(sigma_floor),
        "band": float(band),
    }


@app.get("/ai/predict")
def ai_predict(
    response: Response,
    coin_id: str = Query(..., min_length=1, max_length=128, description="CoinGecko coin id, ex: bitcoin"),
    days: int = Query(90, ge=30, le=365, description="History window"),
    timeframe: str = Query("1d", description="30m | 1h | 4h | 1d"),
):
    tf_norm = timeframe.strip().lower()
    days_eff, interval, h_days = _timeframe_params(tf_norm, days)

    params = {"coin_id": coin_id, "days": days_eff, "interval": interval, "tf": tf_norm}
    cache_key = _make_cache_key("/ai/predict", params)
    throttle_key = f"THROTTLE:/ai/predict:{coin_id}:{tf_norm}:{days_eff}"

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
        response.headers["Retry-After"] = "15"
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

        # 2) history (DAILY)
        url_chart = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
        r2 = session.get(url_chart, params={"vs_currency": "usd", "days": days_eff, "interval": interval}, timeout=25)

        if r2.status_code == 429:
            stale = _cache_get_stale(cache_key, stale_seconds=CACHE_STALE_SECONDS)
            if stale:
                response.headers["X-Cache"] = "STALE-429"
                response.headers["X-Upstream-Status"] = "429"
                ra = r2.headers.get("Retry-After")
                if ra:
                    response.headers["Retry-After"] = ra
                return stale["data"]
            response.headers["Retry-After"] = r2.headers.get("Retry-After") or "30"
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

        if len(closes) < 15:
            return JSONResponse(status_code=502, content={"error": "Not enough history data to predict."})

        rsi = _rsi_14(closes)

        predicted, lo68, hi68, diag = _forecast_next_price(closes, h_days=h_days, rsi=rsi)

        payload = {
            "coin_id": coin_id,
            "timeframe": tf_norm,
            "days_used": days_eff,
            "interval": interval,  # always daily
            "horizon_days": float(h_days),
            "last_price": float(last_price),
            "predicted_price": float(predicted),
            "prediction_interval_68": {"low": float(lo68), "high": float(hi68)},
            "indicators": {"rsi": rsi},
            "model": "statistical-returns-v2",
            "diagnostics": diag,
            "note": "Statistical forecast (daily log-returns + volatility floor). Educational only.",
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
    days: int = Query(90, ge=30, le=365, description="History window"),
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
