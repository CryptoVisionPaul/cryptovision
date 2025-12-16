from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI(title="CryptoVision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# 🔹 SIMPLE DEMO (ce vezi acum)
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

# 🔹 REAL DATA PROXY (pentru tabelul mare)
@app.get("/coins/markets")
def coins_markets(
    vs_currency: str = "usd",
    per_page: int = 50,
    page: int = 1
):
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()
