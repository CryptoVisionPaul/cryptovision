from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI(title="CryptoVision Backend")

# CORS (ok pentru test; mai târziu putem restrânge la domeniul tău GitHub Pages)
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
    r1 = requests.get(url_price, params={"ids": coin_id, "vs_currencies": "usd"}, timeout=20)
    if r1.status_code != 200:
        return {"error": "Failed to fetch price", "coin_id": coin_id, "status": r1.status_code, "body": r1.text}

    price_json = r1.json()
    last_price = (price_json.get(coin_id) or {}).get("usd")
    if last_price is None:
        return {"error": "Unknown coin_id or missing price", "coin_id": coin_id, "raw": price_json}

    # 2) Luăm istoricul pentru RSI simplu (market_chart)
    url_chart = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    r2 = requests.get(url_chart, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=25)
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

    # 3) Predicție DEMO (nu AI real): un mic drift în funcție de RSI
    #    RSI < 40 -> +1.5% (rebound), RSI > 60 -> -1.0% (cooldown), altfel +0.3%
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
        "indicators": {
            "rsi": rsi,
        },
        "note": "Demo endpoint (rule-based). Replace with real AI later.",
    }
