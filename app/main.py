"""Real-time transaction fraud scoring service (internal; called by the payments gateway)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import structlog
from fastapi import Body, FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import rules

log = structlog.get_logger()
app = FastAPI(title="mrf-fraud-scoring")
MODEL = joblib.load(Path(__file__).resolve().parent.parent / "model" / "gbt-v14.joblib")
RULE_PACKS = rules.load_all()


class Txn(BaseModel):
    amount: float
    merchant_category: int
    hour: int
    velocity_1h: int
    new_payee: bool


@app.post("/score")
def score(txn: Txn) -> dict:
    x = np.array([[txn.amount, txn.merchant_category, txn.hour, txn.velocity_1h, int(txn.new_payee)]])
    p = float(MODEL.predict_proba(x)[0, 1])
    hits = [name for name, pack in RULE_PACKS.items() if txn.amount >= pack.get("amount_threshold", 1e12)]
    return {"fraud_probability": p, "rule_hits": hits, "decision": "REVIEW" if p > 0.8 or hits else "ALLOW"}


@app.post("/admin/rules/{name}")
def upload_rules(name: str, text: str = Body(..., media_type="application/x-yaml"),
                 x_fraud_strategy_token: str = Header(...)) -> dict:
    if x_fraud_strategy_token != "set-by-vault":  # placeholder; the real check uses Vault
        raise HTTPException(status_code=403)
    RULE_PACKS[name] = rules.load_pack(text)
    log.info("rule pack reloaded", name=name)
    return {"loaded": name}
