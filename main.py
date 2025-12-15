from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
import os

# -----------------------------
# App
# -----------------------------
app = FastAPI(
    title="CryptoVision Backend",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# -----------------------------
# CORS
# -----------------------------
# Recomandat: setează în Render o variabilă de mediu:
# ALLOWED_ORIGINS=https://cryptovisionpaul.github.io,http://localhost:5500
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if allowed_origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # fallback (merge pentru test)
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class PredictResponse(BaseModel):
    coin_id: str
    last_price: float | None = None
    predicted_price: float | None = None
    indicators: dict
    note: str
    ts_utc: str

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "CryptoVision backend is running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/version")
def version():
    return {
        "name": "cryptovision-backend",
        "version": app.version,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/predict", response_model=PredictResponse)
def predict(
    coin_id: str = Query(..., description="CoinGecko coin id, ex: bitcoin"),
    days: int = Query(90, ge=1, le=365, description="Demo param, 1..365"),
):
    # DEMO: aici urmează să punem logica reală (CoinGecko + indicatori + model AI)
    # Deocamdată returnăm un răspuns stabil ca să poți lega frontend-ul.
    fake_last = None
    fake_pred = None

    indicators = {
        "days": days,
        "rsi": None,
        "macd": None,
        "bbands": None,
        "adx": None,
    }

    return PredictResponse(
        coin_id=coin_id,
        last_price=fake_last,
        predicted_price=fake_pred,
        indicators=indicators,
        note="Demo endpoint. AI + indicatori reali urmează.",
        ts_utc=datetime.now(timezone.utc).isoformat(),
    )
