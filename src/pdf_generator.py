import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from src.config import REPORTS_DIR

URDU_LABELS = {
    "title": "آٹزم پری اسکریننگ رپورٹ",
    "screening_score": "اسکریننگ اسکور",
    "risk_level": "خطرے کی سطح",
    "referral": "رہنمائی",
    "default_prediction": "ابتدائی ماڈل پیش گوئی",
    "screening_prediction": "اسکریننگ پیش گوئی",
    "class_probabilities": "درجہ بندی کے امکانات",
    "disclaimer": "اہم نوٹ: یہ رپورٹ صرف اسکریننگ معاونت کے لیے ہے، طبی تشخیص نہیں۔",
}

EN_LABELS = {
    "title": "Autism Pre-Screening Risk Assessment Report",
    "screening_score": "Screening Score",
    "risk_level": "Score-Based Risk Level",
    "referral": "Referral Interpretation",
    "default_prediction": "ML Default Prediction",
    "screening_prediction": "ML Screening Prediction (Recall-Tuned)",
    "class_probabilities": "Class Probabilities",
    "disclaimer": "Disclaimer: This report is generated for screening support only and is not a medical diagnosis.",
}

FONT_CACHE = {"regular": None, "bold": None}


def normalize_language(language: str | None, report_text: str = "") -> str:
    if str(language or "").strip().lower().startswith("ur"):
        return "ur"
    return "ur" if re.search(r"[\u0600-\u06FF]", report_text or "") else "en"


def register_unicode_fonts() -> tuple[str, str]:
    if FONT_CACHE["regular"] and FONT_CACHE["bold"]:
        return FONT_CACHE["regular"], FONT_CACHE["bold"]

    candidates = [
        (Path(r"C:\Windows\Fonts\tahoma.ttf"), Path(r"C:\Windows\Fonts\tahomabd.ttf")),
        (Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\segoeuib.ttf")),
        (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
    ]

    for regular_path, bold_path in candidates:
        if regular_path.exists() and bold_path.exists():
            regular_name = f"AID_{regular_path.stem}"
            bold_name = f"AID_{bold_path.stem}"
            if regular_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(regular_name, str(regular_path)))
            if bold_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
            FONT_CACHE["regular"] = regular_name
            FONT_CACHE["bold"] = bold_name
            return regular_name, bold_name

    FONT_CACHE["regular"] = "Helvetica"
    FONT_CACHE["bold"] = "Helvetica-Bold"
    return FONT_CACHE["regular"], FONT_CACHE["bold"]


def reshape_rtl_text(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except Exception:
        return text

    return "\n".join(
        get_display(arabic_reshaper.reshape(line)) for line in str(text).splitlines()
    )


def format_paragraph_text(text: str, language: str) -> str:
    rendered = reshape_rtl_text(text) if language == "ur" else str(text)
    return html.escape(rendered).replace("\n", "<br/>")


def generate_pdf_report(
    inference_result: Dict[str, Any],
    report_text: str,
    filename_prefix: str = "autism_prescreen_report",
    language: str = "en",
) -> Path:
    """Generates a PDF report and returns the saved file path."""

    language = normalize_language(language, report_text)
    labels = URDU_LABELS if language == "ur" else EN_LABELS
    regular_font, bold_font = register_unicode_fonts()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORTS_DIR / f"{filename_prefix}_{timestamp}.pdf"

    doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER)
    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "AidTitle",
        parent=base_styles["Title"],
        fontName=bold_font,
        alignment=TA_RIGHT if language == "ur" else TA_LEFT,
    )
    normal_style = ParagraphStyle(
        "AidNormal",
        parent=base_styles["Normal"],
        fontName=regular_font,
        leading=16,
        alignment=TA_RIGHT if language == "ur" else TA_LEFT,
    )
    body_style = ParagraphStyle(
        "AidBody",
        parent=base_styles["BodyText"],
        fontName=regular_font,
        leading=18 if language == "ur" else 16,
        alignment=TA_RIGHT if language == "ur" else TA_LEFT,
    )
    footer_style = ParagraphStyle(
        "AidFooter",
        parent=base_styles["Italic"],
        fontName=regular_font,
        leading=15,
        alignment=TA_RIGHT if language == "ur" else TA_LEFT,
    )

    story = []

    story.append(Paragraph(format_paragraph_text(labels["title"], language), title_style))
    story.append(Spacer(1, 12))

    score = inference_result.get("screening_score", "N/A")
    score_max = inference_result.get("screening_score_max", 24)
    score_risk = inference_result.get("score_risk_level", "N/A")
    referral = inference_result.get("referral_interpretation", "N/A")
    default_pred = inference_result.get("prediction_default", {})
    screening_pred = inference_result.get("prediction_screening", {})
    class_probs = inference_result.get("class_probabilities", {})
    probs_str = ", ".join(f"{k}: {v}" for k, v in class_probs.items())

    summary_lines = [
        f"{labels['screening_score']}: {score}/{score_max}",
        f"{labels['risk_level']}: {score_risk}",
        f"{labels['referral']}: {referral}",
        f"{labels['default_prediction']}: {default_pred.get('predicted_label', 'N/A')}",
        f"{labels['screening_prediction']}: {screening_pred.get('predicted_label', 'N/A')}",
        f"{labels['class_probabilities']}: {probs_str}",
    ]

    story.append(Paragraph(format_paragraph_text("\n".join(summary_lines), language), normal_style))
    story.append(Spacer(1, 14))
    story.append(Paragraph(format_paragraph_text(report_text, language), body_style))
    story.append(Spacer(1, 18))
    story.append(Paragraph(format_paragraph_text(labels["disclaimer"], language), footer_style))

    doc.build(story)
    return pdf_path
