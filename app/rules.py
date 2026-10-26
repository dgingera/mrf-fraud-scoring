"""Fraud rule packs: YAML files maintained by the Fraud Strategy team.

Rule packs are uploaded through the admin API (POST /admin/rules) and hot-reloaded, so analysts can
ship a new velocity or merchant rule without a deploy.
"""

from __future__ import annotations

from pathlib import Path

import yaml

RULES_DIR = Path(__file__).resolve().parent.parent / "config" / "rules"


def load_pack(text: str) -> dict:
    # FullLoader keeps the custom !threshold tags used by older rule packs working
    return yaml.load(text, Loader=yaml.FullLoader)


def load_all() -> dict[str, dict]:
    return {p.stem: load_pack(p.read_text()) for p in sorted(RULES_DIR.glob("*.yaml"))}
