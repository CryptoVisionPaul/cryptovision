from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import time
import threading
from typing import Any, Dict, Tuple

app = FastAPI(title="CryptoVision Backend")

# CORS (ok pentru test; mai târziu restrângem la domeniul tău GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Session HTTP reutilizabil (mai rapid și mai “corect”)
session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "CryptoVisionBackend/1.0 (+https://cryptovisionpaul.github.io)"
})

# ------------------------------------------------------------
# CACHE in-memory (TTL + stale fallback)
# ------------------------------------------------------------
CACHE_TTL_SECONDS = 45  # 30–60 sec e ideal pentru markets
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}
# struct:
# _cache[key] = {
#   "ts": float,
#   "data": Any,
#   "status": int
# }

def _cache_get(key: str):
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if not entry:
            return None
        age = now - entry["ts"]
        if age <= CACHE_TTL_SECONDS:
            return entry  # fresh
        return None

def _cache_get_stale(key: str):
    with _cache_lock:
        return _cache.get(key)

def _cache_set(key: str, data: Any, status: int = 200):
    with _cache_lock:
        _cache[key] = {"ts": time.time(), "data": data, "status": status}

def _make_cache_key(path: str, params: Dict[str, Any]) -> str:
    # key determinist: path + parametrii sortați
    items = sorted((str(k), str(v)) for k, v in params.items())
    return path + "?" + "&".join([f"{k}={v}" for k, v in items])

# ------------------------------------------------------------
# Basic endpoints
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "CryptoVision backend is running"}

@app.get("/health")
def health():
    return {"ok": True}

# ------------------------------------------------------------
# Proxy: /coins/markets (folosit de tabel)
# - Cache TTL
# - Dacă CoinGecko dă 429 -> returnăm ultimul răspuns bun (stale)
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

    # 1) returnăm cache fresh imediat (super rapid)
    cached = _cache_get(cache_key)
    if cached:
        response.headers["X-Cache"] = "HIT"
        response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"
        return cached["data"]

    # 2) altfel, încercăm upstream
    try:
        r = session.get(url, params=params, timeout=20)

        # Dacă upstream e ok
        if r.status_code == 200:
            data = r.json()
            _cache_set(cache_key, data, status=200)
            response.headers["X-Cache"] = "MISS"
            response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"
            return data

        # Dacă upstream e 429 (rate limit) -> servim stale dacă există
        if r.status_code == 429:
            stale = _cache_get_stale(cache_key)
            if stale and stale.get("data") is not None:
                response.headers["X-Cache"] = "STALE"
                response.headers["X-Upstream-Status"] = "429"
                # Retry-After dacă există
                ra = r.headers.get("Retry-After")
                if ra:
                    response.headers["Retry-After"] = ra
                return stale["data"]

            # dacă nu avem cache, returnăm mesaj clar
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Upstream rate limited (CoinGecko). No cached data yet.",
                    "detail": r.text[:300]
                }
            )

        # alte erori upstream
        return JSONResponse(
            status_code=502,
            content={
                "error": "Upstream request failed (CoinGecko).",
                "status_code": r.status_code,
                "detail": r.text[:300],
            }
        )

    except requests.RequestException as e:
        # dacă upstream e down, servim stale dacă există
        stale = _cache_get_stale(cache_key)
        if stale and stale.get("data") is not None:
            response.headers["X-Cache"] = "STALE"
            response.headers["X-Upstream-Status"] = "EXCEPTION"
            return stale["data"]

        return JSONResponse(
            status_code=503,
            content={
                "error": "Upstream request exception (CoinGecko).",
                "detail": str(e),
            }
        )

# ------------------------------------------------------------
# Demo AI: /demo/prediction (folosit de panoul AI din app.html)
# ------------------------------------------------------------
@app.get("/demo/prediction")
def demo_prediction(
    coin_id: str = Query(..., description="CoinGecko coin id, ex: bitcoin"),
    days: int = Query(90, ge=7, le=365, description="History window"),
):
    # 1) Luăm prețul curent
    url_price = f"{COINGECKO_BASE}/simple/price"
    r1 = session.get(url_price, params={"ids": coin_id, "vs_currencies": "usd"}, timeout=20)
    if r1.status_code != 200:
        return {"error": "Failed to fetch price", "coin_id": coin_id, "status": r1.status_code, "body": r1.text}

    price_json = r1.json()
    last_price = (price_json.get(coin_id) or {}).get("usd")
    if last_price is None:
        return {"error": "Unknown coin_id or missing price", "coin_id": coin_id, "raw": price_json}

    # 2) Luăm istoricul pentru RSI simplu (market_chart)
    url_chart = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    r2 = session.get(url_chart, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=25)
    if r2.status_code != 200:
        return {"error": "Failed to fetch history", "coin_id": coin_id, "status": r2.status_code, "body": r2.text}

    prices = (r2.json().get("prices") or [])
    closes = [p[1] for p in prices if isinstance(p, list) and len(p) >= 2]

    # RSI(14) minimal, fără librării externe
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

    # 3) Predicție DEMO (rule-based)
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
