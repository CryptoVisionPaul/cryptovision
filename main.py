from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI(title="CryptoVision Backend")

# CORS – obligatoriu pentru GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COINGECKO_API = "https://api.coingecko.com/api/v3"

@app.get("/")
def root():
    return {"status": "CryptoVision backend is running"}

@app.get("/health")
def health():
    return {"ok": True}

# 🔹 SIMPLE DEMO (3 coins)
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

# 🔹 REAL MARKET DATA (TOP 50)
@app.get("/coins/markets")
def coins_markets(
    vs_currency: str = "usd",
    order: str = "market_cap_desc",
    per_page: int = 50,
    page: int = 1,
    sparkline: bool = False,
    price_change_percentage: str = "1h,24h,7d",
):
    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": order,
        "per_page": per_page,
        "page": page,
        "sparkline": sparkline,
        "price_change_percentage": price_change_percentage,
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()
