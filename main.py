from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI(title="CryptoVision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # poți restrânge mai târziu la GitHub Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


@app.get("/")
def root():
    return {"status": "CryptoVision backend is running"}


@app.get("/health")
def health():
    return {"ok": True}


# ✅ endpoint folosit de UI ca test simplu
@app.get("/coins/simple")
def coins_simple():
    return {
        "source": "backend",
        "coins": [
            {"id": "bitcoin", "price": 89000, "trend": "bullish"},
            {"id": "ethereum", "price": 3150, "trend": "neutral"},
            {"id": "solana", "price": 132, "trend": "bearish"},
        ],
    }


# ✅ IMPORTANT: endpoint pentru tabel (imită CoinGecko /coins/markets)
@app.get("/coins/markets")
def coins_markets(
    vs_currency: str = Query("usd"),
    order: str = Query("market_cap_desc"),
    per_page: int = Query(50, ge=1, le=250),
    page: int = Query(1, ge=1, le=100),
    sparkline: bool = Query(False),
    price_change_percentage: str = Query("1h,24h,7d"),
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

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()
