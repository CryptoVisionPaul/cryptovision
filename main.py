from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = FastAPI(title="CryptoVision Backend", version="1.0.0")

# IMPORTANT:
# Pentru producție, recomand să restrângi allow_origins la:
# ["https://cryptovisionpaul.github.io"]
# Deocamdată lăsăm "*" ca să eliminăm blocaje în timpul testelor.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: închide la domeniul tău după ce totul e stabil
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Sesiune HTTP reutilizabilă (mai performantă decât requests.get repetat)
SESSION = requests.Session()
SESSION.headers.update(
    {
        # User-Agent explicit (ajută uneori la rate-limit / filtrări)
        "User-Agent": "CryptoVisionBackend/1.0 (+https://cryptovisionpaul.github.io)",
        "Accept": "application/json",
    }
)

# ------------------------------------------------------------
# Small cache for /coins/markets (reduce CoinGecko pressure)
# ------------------------------------------------------------
_MARKETS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
MARKETS_CACHE_TTL_SECONDS = 25


# ------------------------------------------------------------
# Error handling (always return JSON, not raw HTML/tracebacks)
# ------------------------------------------------------------
@app.exception_handler(requests.exceptions.Timeout)
def handle_timeout(_: Request, exc: requests.exceptions.Timeout):
    return JSONResponse(
        status_code=504,
        content={"error": "Upstream timeout (CoinGecko). Try again.", "detail": str(exc)},
    )


@app.exception_handler(requests.exceptions.RequestException)
def handle_request_exc(_: Request, exc: requests.exceptions.RequestException):
    return JSONResponse(
        status_code=502,
        content={"error": "Upstream request failed (CoinGecko).", "detail": str(exc)},
    )


@app.exception_handler(Exception)
def handle_generic(_: Request, exc: Exception):
    # Nu expunem stacktrace; doar mesaj generic + detail
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error.", "detail": str(exc)},
    )


# ------------------------------------------------------------
# Basics
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "CryptoVision backend is running", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"ok": True, "service": "cryptovision-backend"}


# ------------------------------------------------------------
# Proxy: /coins/markets  (folosit de tabel)
# ------------------------------------------------------------
@app.get("/coins/markets")
def coins_markets(
    vs_currency: str = "usd",
    order: str = "market_cap_desc",
    per_page: int = 50,
    page: int = 1,
    sparkline: bool = False,
    price_change_percentage: str = "1h,24h,7d",
):
    # Cache (scade latența și presiunea pe CoinGecko)
    now = time.time()
    if (
        _MARKETS_CACHE["data"] is not None
        and (now - float(_MARKETS_CACHE["ts"])) < MARKETS_CACHE_TTL_SECONDS
        and page == 1
        and per_page == 50
        and vs_currency == "usd"
        and order == "market_cap_desc"
    ):
        return _MARKETS_CACHE["data"]

    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": order,
        "per_page": per_page,
        "page": page,
        "sparkline": str(sparkline).lower(),
        "price_change_percentage": price_change_percentage,
    }

    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Populate cache only for the default “main page” query
    if page == 1 and per_page == 50 and vs_currency == "usd" and order == "market_cap_desc":
        _MARKETS_CACHE["ts"] = now
        _MARKETS_CACHE["data"] = data

    return data


# ------------------------------------------------------------
# Compatibility endpoint:
# /coins/simple  (dacă front-end-ul sau cod vechi îl folosește)
# Returnează tot /coins/markets cu parametrii standard.
# ------------------------------------------------------------
@app.get("/coins/simple")
def coins_simple():
    # Este un alias simplu pentru top 50 markets.
    return coins_markets()


# ------------------------------------------------------------
# Demo AI: /demo/prediction (folosit de panoul AI din app.html)
# ------------------------------------------------------------
def rsi_14(closes: list[float]) -> Optional[float]:
    """
    RSI(14) minimal, fără librării externe.
    Calculează pe ultimele 15 închideri (14 diferențe).
    """
    if len(closes) < 15:
        return None

    recent = closes[-15:]
    gains = 0.0
    losses = 0.0

    for i in range(1, len(recent)):
        diff = recent[i] - recent[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += (-diff)

    avg_gain = gains / 14.0
    avg_loss = losses / 14.0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


@app.get("/demo/prediction")
def demo_prediction(
    coin_id: str = Query(..., description="CoinGecko coin id, ex: bitcoin"),
    days: int = Query(90, ge=7, le=365, description="History window"),
):
    # 1) Current price
    url_price = f"{COINGECKO_BASE}/simple/price"
    r1 = SESSION.get(url_price, params={"ids": coin_id, "vs_currencies": "usd"}, timeout=20)
    if r1.status_code != 200:
        return {
            "error": "Failed to fetch price",
            "coin_id": coin_id,
            "status": r1.status_code,
            "body": r1.text[:500],
        }

    price_json = r1.json()
    last_price = (price_json.get(coin_id) or {}).get("usd")
    if last_price is None:
        return {"error": "Unknown coin_id or missing price", "coin_id": coin_id, "raw": price_json}

    # 2) History (market_chart)
    url_chart = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    r2 = SESSION.get(
        url_chart,
        params={"vs_currency": "usd", "days": days, "interval": "daily"},
        timeout=25,
    )
    if r2.status_code != 200:
        return {
            "error": "Failed to fetch history",
            "coin_id": coin_id,
            "status": r2.status_code,
            "body": r2.text[:500],
        }

    prices = (r2.json().get("prices") or [])
    closes = [p[1] for p in prices if isinstance(p, list) and len(p) >= 2 and isinstance(p[1], (int, float))]

    rsi = rsi_14(closes)

    # 3) DEMO prediction (rule-based drift)
    # RSI < 40 -> +1.5% rebound
    # RSI > 60 -> -1.0% cooldown
    # else -> +0.3% neutral drift
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
