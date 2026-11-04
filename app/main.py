"""Real-time transaction fraud scoring service (internal; called by the payments gateway)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import structlog
import torch
from fastapi import Body, FastAPI, Header, HTTPException
from pydantic import BaseModel

from . import rules

log = structlog.get_logger()
app = FastAPI(title="mrf-fraud-scoring")
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL = joblib.load(MODEL_DIR / "gbt-v14.joblib")


class FraudNet(torch.nn.Module):
    """fraud-net v15: small feed-forward network trained by the data-science team (replaces gbt-v14 after shadow)."""

    def __init__(self, n_features: int = 5):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_features, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


def load_shadow_model() -> FraudNet:
    # The weights are exported by the data-science pipeline to the shared model store and synced to model/.
    # weights_only=True is PyTorch's documented safe mode for untrusted checkpoints.
    net = FraudNet()
    net.load_state_dict(torch.load(MODEL_DIR / "fraud-net-v15.pt", weights_only=True))
    return net.eval()


SHADOW = load_shadow_model()
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
    with torch.no_grad():  # shadow mode: scored and logged, not used for the decision yet
        p_v15 = float(torch.sigmoid(SHADOW(torch.tensor(x, dtype=torch.float32)))[0, 0])
    log.info("shadow score", model="fraud-net-v15", p_v14=p, p_v15=p_v15)
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
