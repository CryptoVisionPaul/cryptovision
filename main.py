from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Tuple

import requests
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================================
# App
# ==========================================================
app = FastAPI(title="CryptoVision Backend", version="1.0.0")

# ==========================================================
# CORS
# - Recomandat: setează exact domeniul tău GitHub Pages
# - Dacă vrei temporar "oricine", lasă ["*"] (fără credentials)
# ==========================================================
ALLOWED_ORIGINS = [
    "https://cryptovisionpaul.github.io",
    # "http://localhost:3000",
    # "http://localhost:5173",
]
# Dacă nu vrei să restrângi încă, folosește: ["*"]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # MUST be False when using "*" in allow_origins
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# ==========================================================
# HTTP session (stabil + rapid)
# ==========================================================
session = requests.Session()

retry = Retry(
    total=2,
    backoff_factor=0.6,
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
        "User-Agent": "CryptoVision/1.0 (Render; contact: you)",
    }
)

# ==========================================================
# Cache + Throttle (in-memory)
# ==========================================================
CACHE_TTL_SECONDS = 60          # "fresh" 60s
CACHE_STALE_SECONDS = 15 * 60   # "stale allowed" 15 min
THROTTLE_SECONDS = 45           # minim 45s între upstream call per cheie

# Cache structure:
# _cache[key] = {
#   "ts": float,            # data timestamp
#   "data": Any,            # cached JSON
#   "status": int,          # status that produced this data
#   "last_upstream_ts": float, # last time we hit upstream for this key
# }
_cache: Dict[str, Dict[str, Any]] = {}

def _now() -> float:
    return time.time()

def _make_key(path: str, params: Dict[str, Any]) -> str:
    # key determinist, stabil
    return f"{path}:{json.dumps(params, sort_keys=True, separators=(',', ':'))}"

def _cache_get_fresh(key: str) -> Optional[Dict[str, Any]]:
    obj = _cache.get(key)
    if not obj:
        return None
    age = _now() - float(obj.get("ts", 0.0))
    if age <= CACHE_TTL_SECONDS:
        return obj
    return None

def _cache_get_stale(key: str) -> Optional[Dict[str, Any]]:
    obj = _cache.get(key)
    if not obj:
        return None
    age = _now() - float(obj.get("ts", 0.0))
    if age <= CACHE_STALE_SECONDS:
        return obj
    return None

def _cache_set(key: str, data: Any, status: int = 200) -> None:
    prev = _cache.get(key) or {}
    _cache[key] = {
        "ts": _now(),
        "data": data,
        "status": status,
        "last_upstream_ts": float(prev.get("last_upstream_ts", 0.0)),
    }

def _can_hit_upstream(key: str) -> bool:
    obj = _cache.get(key)
    if not obj:
        return True
    last = float(obj.get("last_upstream_ts", 0.0) or 0.0)
    return (_now() - last) >= THROTTLE_SECONDS

def _mark_upstream_hit(key: str) -> None:
    obj = _cache.get(key)
    if not obj:
        _cache[key] = {"ts": 0.0, "data": None, "status": 0, "last_upstream_ts": _now()}
    else:
        obj["last_upstream_ts"] = _now()

def _set_cache_headers(response: Response, *, cache_state: str, upstream_status: Optional[str] = None) -> None:
    # private cache; SWR = browser poate folosi "stale" un pic fără să refacă imediat
    response.headers["Cache-Control"] = f"private, max-age={CACHE_TTL_SECONDS}, stale-while-revalidate={CACHE_TTL_SECONDS}"
    response.headers["X-Cache"] = cache_state
    if upstream_status is not None:
        response.headers["X-Upstream-Status"] = upstream_status

def _safe_json_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text[:500]}

def _upstream_get_json(
    *,
    url: str,
    params: Dict[str, Any],
    timeout: int,
) -> Tuple[int, Any, str]:
    """
    Return (status_code, data_json_or_error, raw_text_snippet)
    """
    try:
        r = session.get(url, params=params, timeout=timeout)
        status = int(r.status_code)

        if status == 200:
            return status, r.json(), ""

        # for non-200, keep small snippet
        raw = (r.text or "")[:500]
        # some CoinGecko errors are JSON inside text
        parsed = _safe_json_parse(r.text or "")
        return status, parsed, raw

    except requests.RequestException as e:
        return 0, {"error": "request_exception", "detail": str(e)}, str(e)[:500]

# ==========================================================
# Routes
# ==========================================================
@app.get("/")
def root():
    return {"status": "CryptoVision backend is running"}

@app.get("/health")
def health():
    return {"ok": True}

# ------------------------------------------------------------
# Proxy: /coins/markets  (Top 50)
# - cache fresh 60s
# - stale up to 15 min if 429/exception
# - throttle per unique params
# ------------------------------------------------------------
@app.get("/coins/markets")
def coins_markets(
    response: Response,
    vs_currency: str = "usd",
    order: str = "market_cap_desc",
    per_page: int = 50,
    page: int = 1,
    sparkline: bool = False,
    price_change_percentage: str = "1h,24h,7d",
):
    path = "/coins/markets"
    url = f"{COINGECKO_BASE}{path}"

    params = {
        "vs_currency": vs_currency,
        "order": order,
        "per_page": int(per_page),
        "page": int(page),
        "sparkline": "true" if sparkline else "false",
        "price_change_percentage": price_change_percentage,
    }

    cache_key = _make_key(path, params)
    throttle_key = "THROTTLE:" + cache_key

    # fallback last good (independent of params)
    last_good_key = "LAST_GOOD:/coins/markets"

    # 1) Fresh cache
    cached = _cache_get_fresh(cache_key)
    if cached:
        _set_cache_headers(response, cache_state="HIT")
        return cached["data"]

    # 2) Throttle guard (per params)
    if not _can_hit_upstream(throttle_key):
        stale = _cache_get_stale(cache_key)
        if stale:
            _set_cache_headers(response, cache_state="STALE-THROTTLE")
            return stale["data"]

        stale_last_good = _cache_get_stale(last_good_key)
        if stale_last_good:
            _set_cache_headers(response, cache_state="STALE-LAST-GOOD-THROTTLE")
            return stale_last_good["data"]

        return JSONResponse(
            status_code=429,
            content={"error": "Throttled to protect upstream. No cached data yet."},
        )

    _mark_upstream_hit(throttle_key)

    # 3) Upstream call
    status, data, raw = _upstream_get_json(url=url, params=params, timeout=20)

    # 3a) success
    if status == 200 and isinstance(data, list):
        _cache_set(cache_key, data, status=200)
        _cache_set(last_good_key, data, status=200)
        _set_cache_headers(response, cache_state="MISS", upstream_status="200")
        return data

    # 3b) rate limited -> serve stale
    if status == 429:
        stale = _cache_get_stale(cache_key)
        if stale:
            _set_cache_headers(response, cache_state="STALE-EXACT", upstream_status="429")
            return stale["data"]

        stale_last_good = _cache_get_stale(last_good_key)
        if stale_last_good:
            _set_cache_headers(response, cache_state="STALE-LAST-GOOD", upstream_status="429")
            return stale_last_good["data"]

        return JSONResponse(
            status_code=429,
            content={
                "error": "Upstream rate limited (CoinGecko). No cached data yet.",
                "detail": data if isinstance(data, dict) else {"raw": raw},
            },
        )

    # 3c) upstream exception or other errors -> serve last good if possible
    stale_last_good = _cache_get_stale(last_good_key)
    if stale_last_good:
        _set_cache_headers(response, cache_state="STALE-LAST-GOOD-UPSTREAM-ERROR", upstream_status=str(status or "EXCEPTION"))
        return stale_last_good["data"]

    return JSONResponse(
        status_code=502,
        content={
            "error": "Upstream request failed (CoinGecko).",
            "status_code": status or 0,
            "detail": data if isinstance(data, dict) else {"raw": raw},
        },
    )

# ------------------------------------------------------------
# Proxy: /coins/{coin_id}/market_chart
# - folosit pentru chart în UI (ideal prin backend ca să nu lovești 429 din browser)
# - cache: fresh 60s, stale 15 min
# - throttle per params
# ------------------------------------------------------------
@app.get("/coins/{coin_id}/market_chart")
def market_chart(
    coin_id: str,
    response: Response,
    vs_currency: str = "usd",
    days: int = 7,
    interval: str = "hourly",
):
    path = f"/coins/{coin_id}/market_chart"
    url = f"{COINGECKO_BASE}{path}"

    params = {
        "vs_currency": vs_currency,
        "days": int(days),
        "interval": interval,
    }

    cache_key = _make_key(path, params)
    throttle_key = "THROTTLE:" + cache_key
    last_good_key = f"LAST_GOOD:{path}"

    cached = _cache_get_fresh(cache_key)
    if cached:
        _set_cache_headers(response, cache_state="HIT")
        return cached["data"]

    if not _can_hit_upstream(throttle_key):
        stale = _cache_get_stale(cache_key)
        if stale:
            _set_cache_headers(response, cache_state="STALE-THROTTLE")
            return stale["data"]

        stale_last = _cache_get_stale(last_good_key)
        if stale_last:
            _set_cache_headers(response, cache_state="STALE-LAST-GOOD-THROTTLE")
            return stale_last["data"]

        return JSONResponse(
            status_code=429,
            content={"error": "Throttled to protect upstream. No cached data yet."},
        )

    _mark_upstream_hit(throttle_key)

    status, data, raw = _upstream_get_json(url=url, params=params, timeout=25)

    if status == 200 and isinstance(data, dict):
        _cache_set(cache_key, data, status=200)
        _cache_set(last_good_key, data, status=200)
        _set_cache_headers(response, cache_state="MISS", upstream_status="200")
        return data

    if status == 429:
        stale = _cache_get_stale(cache_key)
        if stale:
            _set_cache_headers(response, cache_state="STALE-EXACT", upstream_status="429")
            return stale["data"]

        stale_last = _cache_get_stale(last_good_key)
        if stale_last:
            _set_cache_headers(response, cache_state="STALE-LAST-GOOD", upstream_status="429")
            return stale_last["data"]

        return JSONResponse(
            status_code=429,
            content={
                "error": "Upstream rate limited (CoinGecko). No cached data yet.",
                "detail": data if isinstance(data, dict) else {"raw": raw},
            },
        )

    stale_last = _cache_get_stale(last_good_key)
    if stale_last:
        _set_cache_headers(response, cache_state="STALE-LAST-GOOD-UPSTREAM-ERROR", upstream_status=str(status or "EXCEPTION"))
        return stale_last["data"]

    return JSONResponse(
        status_code=502,
        content={
            "error": "Upstream request failed (CoinGecko).",
            "status_code": status or 0,
            "detail": data if isinstance(data, dict) else {"raw": raw},
        },
    )

# ------------------------------------------------------------
# Helper (cached): /simple/price
# - folosit intern de demo_prediction
# ------------------------------------------------------------
def _get_simple_price_cached(coin_id: str) -> Tuple[Optional[float], Optional[dict]]:
    path = "/simple/price"
    url = f"{COINGECKO_BASE}{path}"
    params = {"ids": coin_id, "vs_currencies": "usd"}

    cache_key = _make_key("INTERNAL" + path, params)
    throttle_key = "THROTTLE:" + cache_key
    last_good_key = f"LAST_GOOD:INTERNAL{path}:{coin_id}"

    cached = _cache_get_fresh(cache_key)
    if cached and isinstance(cached["data"], dict):
        price = (cached["data"].get(coin_id) or {}).get("usd")
        return float(price) if price is not None else None, None

    if _can_hit_upstream(throttle_key):
        _mark_upstream_hit(throttle_key)
        status, data, _raw = _upstream_get_json(url=url, params=params, timeout=20)
        if status == 200 and isinstance(data, dict):
            _cache_set(cache_key, data, status=200)
            _cache_set(last_good_key, data, status=200)
            price = (data.get(coin_id) or {}).get("usd")
            return float(price) if price is not None else None, None

        # if failed, try stale
        stale_last = _cache_get_stale(last_good_key)
        if stale_last and isinstance(stale_last["data"], dict):
            price = (stale_last["data"].get(coin_id) or {}).get("usd")
            return float(price) if price is not None else None, {"warning": "stale_price_used", "upstream_status": status, "detail": data}

        return None, {"error": "price_fetch_failed", "upstream_status": status, "detail": data}

    # throttled -> stale
    stale_last = _cache_get_stale(last_good_key)
    if stale_last and isinstance(stale_last["data"], dict):
        price = (stale_last["data"].get(coin_id) or {}).get("usd")
        return float(price) if price is not None else None, {"warning": "stale_price_used_throttle"}
    return None, {"error": "price_throttled_no_cache"}

# ------------------------------------------------------------
# Helper (cached): market_chart daily for RSI
# ------------------------------------------------------------
def _get_market_chart_daily_cached(coin_id: str, days: int) -> Tuple[Optional[list], Optional[dict]]:
    path = f"/coins/{coin_id}/market_chart"
    url = f"{COINGECKO_BASE}{path}"

    params = {"vs_currency": "usd", "days": int(days), "interval": "daily"}

    cache_key = _make_key("INTERNAL" + path, params)
    throttle_key = "THROTTLE:" + cache_key
    last_good_key = f"LAST_GOOD:INTERNAL{path}:{days}"

    cached = _cache_get_fresh(cache_key)
    if cached and isinstance(cached["data"], dict):
        prices = cached["data"].get("prices") or []
        return prices, None

    if _can_hit_upstream(throttle_key):
        _mark_upstream_hit(throttle_key)
        status, data, _raw = _upstream_get_json(url=url, params=params, timeout=25)
        if status == 200 and isinstance(data, dict):
            _cache_set(cache_key, data, status=200)
            _cache_set(last_good_key, data, status=200)
            prices = data.get("prices") or []
            return prices, None

        stale_last = _cache_get_stale(last_good_key)
        if stale_last and isinstance(stale_last["data"], dict):
            prices = stale_last["data"].get("prices") or []
            return prices, {"warning": "stale_history_used", "upstream_status": status, "detail": data}

        return None, {"error": "history_fetch_failed", "upstream_status": status, "detail": data}

    stale_last = _cache_get_stale(last_good_key)
    if stale_last and isinstance(stale_last["data"], dict):
        prices = stale_last["data"].get("prices") or []
        return prices, {"warning": "stale_history_used_throttle"}
    return None, {"error": "history_throttled_no_cache"}

# ------------------------------------------------------------
# Demo AI: /demo/prediction
# - acum are cache implicit pentru price + history
# ------------------------------------------------------------
@app.get("/demo/prediction")
def demo_prediction(
    coin_id: str = Query(..., description="CoinGecko coin id, ex: bitcoin"),
    days: int = Query(90, ge=7, le=365, description="History window for RSI"),
):
    coin_id = coin_id.strip().lower()

    last_price, price_meta = _get_simple_price_cached(coin_id)
    if last_price is None:
        return JSONResponse(
            status_code=502,
            content={"error": "Failed to fetch price", "coin_id": coin_id, "detail": price_meta},
        )

    prices, hist_meta = _get_market_chart_daily_cached(coin_id, days)
    if prices is None:
        return JSONResponse(
            status_code=502,
            content={"error": "Failed to fetch history", "coin_id": coin_id, "detail": hist_meta},
        )

    closes = [p[1] for p in prices if isinstance(p, list) and len(p) >= 2]

    def rsi_14(series: list) -> Optional[float]:
        if len(series) < 15:
            return None
        gains = []
        losses = []
        # last 15 closes -> 14 diffs
        recent = series[-15:]
        for i in range(1, 15):
            diff = recent[i] - recent[i - 1]
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

    rsi = rsi_14(closes)

    # Demo drift rule-based
    if rsi is None:
        drift = 0.003
        label = "demo-neutral"
    elif rsi < 40:
        drift = 0.015
        label = "demo-rebound"
    elif rsi > 60:
        drift = -0.010
        label = "demo-cooldown"
    else:
        drift = 0.003
        label = "demo-neutral"

    predicted_price = float(last_price) * (1.0 + drift)

    meta = {}
    if price_meta:
        meta["price"] = price_meta
    if hist_meta:
        meta["history"] = hist_meta

    return {
        "coin_id": coin_id,
        "model": label,
        "days": days,
        "last_price": float(last_price),
        "predicted_price": float(predicted_price),
        "indicators": {"rsi": rsi},
        "note": "Demo endpoint (rule-based). Replace with real AI later.",
        "meta": meta or None,
    }
