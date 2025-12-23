from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = FastAPI(title="CryptoVision Backend")

# CORS (pentru test e OK; ulterior îl restrângem la domeniul tău)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# -------------------------------------------------------------------
# HTTP session (mai rapid + mai stabil decât requests.get repetat)
# -------------------------------------------------------------------
session = requests.Session()
retry = Retry(
    total=2,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
session.mount("https://", adapter)
session.mount("http://", adapter)
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "CryptoVision/1.0 (Render; contact: you)"
})

# -------------------------------------------------------------------
# Cache in-memory (Render free: se resetează la redeploy/restart — OK)
# Strategie:
# - TTL: cât considerăm “fresh”
# - STALE: cât acceptăm să servim “ultimul bun” dacă upstream e 429
# - Throttle: nu lovim CoinGecko mai des decât la X sec pe același endpoint
# -------------------------------------------------------------------
CACHE_TTL_SECONDS = 60          # fresh 60s
CACHE_STALE_SECONDS = 15 * 60   # acceptăm stale 15 min dacă e 429
THROTTLE_SECONDS = 45           # minim 45s între request-uri upstream

_cache = {}  # key -> {"ts": float, "data": any, "status": int, "last_upstream_ts": float}

def _now() -> float:
    return time.time()

def _cache_get(key: str):
    obj = _cache.get(key)
    if not obj:
        return None
    age = _now() - obj["ts"]
    if age <= CACHE_TTL_SECONDS:
        return obj
    return None

def _cache_get_stale(key: str):
    obj = _cache.get(key)
    if not obj:
        return None
    age = _now() - obj["ts"]
    if age <= CACHE_STALE_SECONDS:
        return obj
    return None

def _cache_set(key: str, data, status: int = 200):
    _cache[key] = {
        "ts": _now(),
        "data": data,
        "status": status,
        "last_upstream_ts": _cache.get(key, {}).get("last_upstream_ts", 0.0),
    }

def _make_cache_key(path: str, params: dict) -> str:
    # cheie deterministă
    return f"{path}:{json.dumps(params, sort_keys=True)}"

def _can_hit_upstream(throttle_key: str) -> bool:
    obj = _cache.get(throttle_key)
    if not obj:
        return True
    last = obj.get("last_upstream_ts", 0.0) or 0.0
    return (_now() - last) >= THROTTLE_SECONDS

def _mark_upstream_hit(throttle_key: str):
    obj = _cache.get(throttle_key)
    if not obj:
        _cache[throttle_key] = {"ts": 0.0, "data": None, "status": 0, "last_upstream_ts": _now()}
    else:
        obj["last_upstream_ts"] = _now()

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "CryptoVision backend is running"}

@app.get("/health")
def health():
    return {"ok": True}

# ------------------------------------------------------------
# Proxy: /coins/markets
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

    # last-good global: dacă query-ul curent pică, servim ultima listă bună
    last_good_key = "LAST_GOOD:/coins/markets"
    throttle_key = "THROTTLE:/coins/markets"

    # 1) Dacă avem fresh cache pentru query exact -> HIT
    cached = _cache_get(cache_key)
    if cached:
        response.headers["X-Cache"] = "HIT"
        response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"
        return cached["data"]

    # 2) Dacă nu putem lovi upstream (throttle), încercăm stale (query / last-good)
    if not _can_hit_upstream(throttle_key):
        stale_exact = _cache_get_stale(cache_key)
        if stale_exact:
            response.headers["X-Cache"] = "STALE-THROTTLE"
            return stale_exact["data"]

        stale_last_good = _cache_get_stale(last_good_key)
        if stale_last_good:
            response.headers["X-Cache"] = "STALE-LAST-GOOD-THROTTLE"
            return stale_last_good["data"]

        return JSONResponse(
            status_code=429,
            content={"error": "Throttled to protect upstream. No cached data yet."}
        )

    # Marcam attempt upstream
    _mark_upstream_hit(throttle_key)

    # 3) Upstream call
    try:
        r = session.get(url, params=params, timeout=20)

        # 3a) SUCCESS
        if r.status_code == 200:
            data = r.json()
            _cache_set(cache_key, data, status=200)
            _cache_set(last_good_key, data, status=200)

            response.headers["X-Cache"] = "MISS"
            response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"
            return data

        # 3b) RATE LIMITED: servim stale dacă există
        if r.status_code == 429:
            stale_exact = _cache_get_stale(cache_key)
            if stale_exact:
                response.headers["X-Cache"] = "STALE-EXACT"
                response.headers["X-Upstream-Status"] = "429"
                ra = r.headers.get("Retry-After")
                if ra:
                    response.headers["Retry-After"] = ra
                return stale_exact["data"]

            stale_last_good = _cache_get_stale(last_good_key)
            if stale_last_good:
                response.headers["X-Cache"] = "STALE-LAST-GOOD"
                response.headers["X-Upstream-Status"] = "429"
                ra = r.headers.get("Retry-After")
                if ra:
                    response.headers["Retry-After"] = ra
                return stale_last_good["data"]

            # Dacă nu există nimic cached încă
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Upstream rate limited (CoinGecko). No cached data yet.",
                    "detail": r.text[:500],
                },
            )

        # 3c) Other upstream errors
        return JSONResponse(
            status_code=502,
            content={
                "error": "Upstream request failed (CoinGecko).",
                "status_code": r.status_code,
                "detail": r.text[:500],
            },
        )

    except requests.RequestException as e:
        # 4) Dacă upstream e down, încercăm stale
        stale_last_good = _cache_get_stale(last_good_key)
        if stale_last_good:
            response.headers["X-Cache"] = "STALE-LAST-GOOD-EXCEPTION"
            response.headers["X-Upstream-Status"] = "EXCEPTION"
            return stale_last_good["data"]

        return JSONResponse(
            status_code=503,
            content={"error": "Upstream request exception (CoinGecko).", "detail": str(e)},
        )

# ------------------------------------------------------------
# Demo AI: /demo/prediction (cum ai avut)
# ------------------------------------------------------------
@app.get("/demo/prediction")
def demo_prediction(
    coin_id: str = Query(..., description="CoinGecko coin id, ex: bitcoin"),
    days: int = Query(90, ge=7, le=365, description="History window"),
):
    url_price = f"{COINGECKO_BASE}/simple/price"
    r1 = session.get(url_price, params={"ids": coin_id, "vs_currencies": "usd"}, timeout=20)
    if r1.status_code != 200:
        return {"error": "Failed to fetch price", "coin_id": coin_id, "status": r1.status_code, "body": r1.text}

    price_json = r1.json()
    last_price = (price_json.get(coin_id) or {}).get("usd")
    if last_price is None:
        return {"error": "Unknown coin_id or missing price", "coin_id": coin_id, "raw": price_json}

    url_chart = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    r2 = session.get(url_chart, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=25)
    if r2.status_code != 200:
        return {"error": "Failed to fetch history", "coin_id": coin_id, "status": r2.status_code, "body": r2.text}

    prices = (r2.json().get("prices") or [])
    closes = [p[1] for p in prices if isinstance(p, list) and len(p) >= 2]

    def rsi_14(series):
        if len(series) < 15:
            return None
        gains = []
        losses = []
        for i in range(1, 15):
            diff = series[-(15 - i)] - series[-(16 - i)]
            if diff >= 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-diff)
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi_14(closes)

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

    return {
        "coin_id": coin_id,
        "model": label,
        "days": days,
        "last_price": float(last_price),
        "predicted_price": float(predicted_price),
        "indicators": {"rsi": rsi},
        "note": "Demo endpoint (rule-based). Replace with real AI later.",
    }
