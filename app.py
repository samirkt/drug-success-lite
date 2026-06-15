"""Minimal FastAPI server for the molecule+disease approval-probability tool.

    uv run uvicorn app:app --port 8080

Serves the single-page UI at `/` and scores one (SMILES, ICD-10) pair at POST `/predict`.
Same-origin, so no CORS needed.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dsm.resolve import resolve_disease, resolve_drug
from dsm.serve import load_predictor, predict_one

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Drug approval probability — md model")


class PredictRequest(BaseModel):
    smiles: str = ""
    icd_codes: list[str] = []


@app.on_event("startup")
def _warmup() -> None:
    load_predictor()  # build/load the artifact once at startup


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "predict.html")


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        return predict_one(req.smiles, req.icd_codes)
    except Exception as exc:  # surface featurization/model errors as 400s
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/resolve/drug")
def resolve_drug_ep(name: str) -> dict:
    try:
        return resolve_drug(name)
    except Exception as exc:  # external API/network failure
        raise HTTPException(status_code=502, detail=f"ChEMBL lookup failed: {exc}")


@app.get("/resolve/disease")
def resolve_disease_ep(name: str) -> dict:
    try:
        return resolve_disease(name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ICD lookup failed: {exc}")
