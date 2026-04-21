"""
FastAPI server: serves the static website and exposes ML / report endpoints.
Run from project root:
  python -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import predict_autism_risk
from src.llm_report_groq import generate_risk_report
from src.pdf_generator import generate_pdf_report
from src.face_screening import predict_face_binary

# Prefer the JS SPA that calls this FastAPI backend directly.
# Keep the legacy Frontend folder as a fallback so older repo states still run.
WEB_DIR = ROOT / "web" / "public"
if not WEB_DIR.is_dir():
    WEB_DIR = ROOT / "Frontend"

app = FastAPI(title="Autism Pre-Screening API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScreenPayload(BaseModel):
    age_mons: int = Field(..., ge=1, le=144)
    gender: Optional[str] = None
    sex: Optional[str] = None
    jaundice: str
    family_mem_with_asd: str
    qchat_answers: Dict[str, str]
    mchat_answers: Dict[str, str]

    class Config:
        json_schema_extra = {
            "example": {
                "age_mons": 24,
                "gender": "male",
                "jaundice": "no",
                "family_mem_with_asd": "no",
                "qchat_answers": {1: "A", 2: "B"},
                "mchat_answers": {11: "Yes", 12: "No"},
            }
        }


class ReportLLMPayload(BaseModel):
    inference_result: dict
    language: Optional[str] = None


class PhotoScreenPayload(BaseModel):
    image: str


class ReportPDFPayload(BaseModel):
    inference_result: dict
    report_text: str
    language: Optional[str] = None


@app.post("/api/face/predict")
async def api_face_predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    try:
        from PIL import Image

        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        result = predict_face_binary(pil)
        return JSONResponse(result)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Face prediction failed: {e}")


@app.post("/api/photo/screen")
def api_photo_screen(body: PhotoScreenPayload):
    if not body.image or not body.image.strip():
        raise HTTPException(400, "Image is required.")
    try:
        from PIL import Image

        encoded = body.image.strip()
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        raw = base64.b64decode(encoded)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        result = predict_face_binary(pil)
        return {
            "result": "autistic" if result.get("is_autistic") else "non-autistic",
            **result,
        }
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Photo screening failed: {e}")


@app.post("/api/screen/predict")
def api_screen_predict(body: ScreenPayload):
    try:
        gender = body.gender or body.sex
        payload = {
            "age_mons": body.age_mons,
            "gender": gender,
            "jaundice": body.jaundice,
            "family_mem_with_asd": body.family_mem_with_asd,
            "qchat_answers": {int(k): v for k, v in body.qchat_answers.items()},
            "mchat_answers": {int(k): v for k, v in body.mchat_answers.items()},
        }
        result = predict_autism_risk(payload)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/report/llm")
def api_report_llm(body: ReportLLMPayload):
    try:
        text = generate_risk_report(body.inference_result, language=body.language or "en")
        return {"report_text": text}
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")


@app.post("/api/report/pdf")
def api_report_pdf(body: ReportPDFPayload):
    try:
        path = generate_pdf_report(
            body.inference_result,
            body.report_text,
            language=body.language or "en",
        )
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=path.name,
        )
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")


@app.get("/api/health")
def api_health():
    return {"ok": True}


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
