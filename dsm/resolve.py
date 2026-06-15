"""Name resolution for the web tool: drug name -> SMILES (ChEMBL) and disease name -> ICD-10-CM
candidates (NLM Clinical Tables). Stdlib-only HTTP (no new deps), small in-memory cache, polite
User-Agent. Network failures raise; the API layer turns them into a 502.

Both are optional convenience layers — the model is still driven by raw SMILES + ICD codes.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from functools import lru_cache

logger = logging.getLogger(__name__)

_UA = "drug-success-lite/0.1 (open-source research tool)"
_TIMEOUT = 20

CHEMBL_SEARCH = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
NLM_ICD10CM = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"


def _get_json(url: str, params: dict):
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


@lru_cache(maxsize=512)
def resolve_drug(name: str, limit: int = 5) -> dict:
    """Drug name -> best ChEMBL SMILES match (+ a few candidates). `smiles` is None for entities
    with no small-molecule structure (e.g. biologics)."""
    name = (name or "").strip()
    if not name:
        return {"query": name, "smiles": None, "candidates": []}

    data = _get_json(CHEMBL_SEARCH, {"q": name, "limit": limit})
    candidates = []
    for m in data.get("molecules", []):
        smiles = (m.get("molecule_structures") or {}).get("canonical_smiles")
        candidates.append({
            "chembl_id": m.get("molecule_chembl_id"),
            "pref_name": m.get("pref_name"),
            "smiles": smiles,
        })
    best = next((c for c in candidates if c["smiles"]), None)
    return {
        "query": name,
        "smiles": best["smiles"] if best else None,
        "chembl_id": best["chembl_id"] if best else None,
        "pref_name": best["pref_name"] if best else None,
        "candidates": candidates,
    }


@lru_cache(maxsize=512)
def resolve_disease(name: str, max_list: int = 7) -> dict:
    """Disease name -> ICD-10-CM candidate [{code, name}] from NLM Clinical Tables. Matches against
    official ICD names, so phrasing matters ('breast cancer' won't match 'malignant neoplasm of
    breast') — candidates are returned for the user to pick/refine."""
    name = (name or "").strip()
    if not name:
        return {"query": name, "candidates": []}

    # sf=code,name is required to search by disease name (default searches code only).
    res = _get_json(NLM_ICD10CM, {"terms": name, "sf": "code,name", "maxList": max_list})
    pairs = res[3] if len(res) > 3 and res[3] else []
    return {"query": name, "candidates": [{"code": c, "name": n} for c, n in pairs]}
