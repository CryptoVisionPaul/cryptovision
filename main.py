from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CryptoVision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "CryptoVision backend is running"}

@app.get("/health")
def health():
    return {"ok": True}

# 🔹 TEST ENDPOINT – coins (mock data)
@app.get("/coins/simple")
def get_simple_coins():
    return {
        "source": "backend",
        "coins": [
            {
                "id": "bitcoin",
                "price": 89000,
                "trend": "bullish"
            },
            {
                "id": "ethereum",
                "price": 3150,
                "trend": "neutral"
            },
            {
                "id": "solana",
                "price": 132,
                "trend": "bearish"
            }
        ]
    }
