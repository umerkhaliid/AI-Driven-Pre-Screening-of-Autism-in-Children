import sys
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from src.face_screening import (
    dev_bypass_face_screening,
    predict_face_binary_or_bypass,
    resolve_face_classifier_path,
)
from src.inference import predict_autism_risk
from src.llm_report_groq import generate_risk_report
from src.pdf_generator import generate_pdf_report
from src.config import FACE_CLASSIFIER_DEFAULT_PATH


st.set_page_config(
    page_title="Autism Pre-Screening Tool",
    layout="centered",
)

st.title("Autism Pre-Screening Tool")
st.caption(
    "Photo-based triage, then Q-CHAT-10 + M-CHAT-R screening. Not a medical diagnosis."
)

if "face_result" not in st.session_state:
    st.session_state["face_result"] = None
if "inference_result" not in st.session_state:
    st.session_state["inference_result"] = None
if "report_text" not in st.session_state:
    st.session_state["report_text"] = None
if "pdf_path" not in st.session_state:
    st.session_state["pdf_path"] = None


def clear_downstream_after_face_change():
    st.session_state["inference_result"] = None
    st.session_state["report_text"] = None
    st.session_state["pdf_path"] = None


def allow_questionnaire() -> bool:
    if dev_bypass_face_screening():
        return True
    r = st.session_state.get("face_result")
    return r is not None and bool(r.get("is_autistic"))


# ---------------------------------------------------------------------------
# Step 1: Face image — binary Autistic / Non_Autistic gate
# ---------------------------------------------------------------------------
st.subheader("Step 1: Photo screening (face-based)")
st.write(
    "Upload a clear photo of your child’s face. The app runs a transfer-learning "
    "image classifier first. **Only if this step indicates Autistic** will the "
    "questionnaires and full screening continue."
)

if dev_bypass_face_screening():
    st.info(
        "Developer bypass is on (`DEV_BYPASS_FACE_SCREENING`): the photo model is "
        "skipped and questionnaires are always available."
    )
else:
    model_path = resolve_face_classifier_path()
    if model_path is None:
        st.warning(
            "No face classifier weights found. Add your trained model file to:\n\n"
            f"`{FACE_CLASSIFIER_DEFAULT_PATH}`\n\n"
            "or set the environment variable `FACE_CLASSIFIER_MODEL_PATH` to the "
            "full path of your `.pth` checkpoint (PyTorch EfficientNet-B0)."
        )

col_a, col_b = st.columns(2)
with col_a:
    uploaded = st.file_uploader(
        "Child photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Front-facing, well-lit photos work best.",
    )
with col_b:
    if st.button("Reset screening (start over)", type="secondary"):
        st.session_state["face_result"] = None
        clear_downstream_after_face_change()
        st.rerun()

if not dev_bypass_face_screening() and st.button("Run photo screening", type="primary"):
    if uploaded is None:
        st.error("Please upload a photo first.")
    else:
        try:
            pil = Image.open(BytesIO(uploaded.getvalue()))
            with st.spinner("Analyzing photo…"):
                st.session_state["face_result"] = predict_face_binary_or_bypass(pil)
            clear_downstream_after_face_change()
            st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Photo screening failed: {e}")

if dev_bypass_face_screening():
    st.session_state["face_result"] = predict_face_binary_or_bypass(None)

face_result = st.session_state.get("face_result")

if face_result is not None:
    st.divider()
    st.write("**Photo screening result**")
    if face_result.get("dev_bypass"):
        st.caption("Synthetic result (developer bypass).")
    else:
        st.caption(f"Model: `{face_result.get('model_path')}`")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted class", face_result["predicted_label"])
    with c2:
        st.metric("Continue to questionnaires", "Yes" if face_result["is_autistic"] else "No")
    probs = face_result.get("probabilities") or {}
    for label, p in probs.items():
        st.progress(min(max(p, 0.0), 1.0), text=f"{label}: {p:.1%}")

    if not face_result["is_autistic"]:
        st.error(
            "The photo-based step did **not** flag this case for the behavioural "
            "questionnaires in this workflow. The process stops here. If you still "
            "have developmental concerns, speak with a qualified professional."
        )
        st.stop()

    st.success(
        "Photo screening step complete. You can continue with child details and "
        "the questionnaires below."
    )

elif not dev_bypass_face_screening():
    st.info("Run photo screening above to continue.")
    st.stop()

st.divider()

if not allow_questionnaire():
    st.stop()

# ---------------------------------------------------------------------------
# Step 2: Child Information
# ---------------------------------------------------------------------------
st.subheader("Step 2: Child Information")

age_mons = st.number_input("Age (in months)", min_value=12, max_value=60, value=24)
gender = st.selectbox("Gender", ["male", "female"])
jaundice = st.selectbox("Jaundice", ["no", "yes"])
family_mem_with_asd = st.selectbox("Family ASD", ["no", "yes"])

st.divider()


# ---------------------------------------------------------------------------
# Step 3: Q-CHAT-10 Questions (A-E answers)
# ---------------------------------------------------------------------------
st.subheader("Step 3: Q-CHAT-10 Questionnaire")
st.write("Select the best answer (A-E) for each question.")

QCHAT_QUESTIONS = {
    1: "Does your child look at you when you call his/her name?",
    2: "How easy is it for you to get eye contact with your child?",
    3: "Does your child point to indicate that s/he wants something?",
    4: "Does your child point to share interest with you?",
    5: "Does your child pretend? (e.g. care for dolls, talk on toy phone)",
    6: "Does your child follow where you're looking?",
    7: "Does your child comfort upset family members?",
    8: "Would you describe your child's first words as typical?",
    9: "Does your child use simple gestures? (e.g. wave goodbye)",
    10: "Does your child stare at nothing with no apparent purpose?",
}

ANSWER_OPTIONS_AE = ["A", "B", "C", "D", "E"]

qchat_answers = {}
for i, question in QCHAT_QUESTIONS.items():
    qchat_answers[i] = st.selectbox(
        f"Q{i}: {question}",
        ANSWER_OPTIONS_AE,
        key=f"qchat_{i}",
    )

st.divider()


# ---------------------------------------------------------------------------
# Step 4: M-CHAT-R Questions (Yes/No answers)
# ---------------------------------------------------------------------------
st.subheader("Step 4: M-CHAT-R Questionnaire")
st.write("Answer Yes or No for each question.")

MCHAT_QUESTIONS = {
    11: "If you point at something across the room, does your child look at it?",
    12: "Have you ever wondered if your child might be deaf?",
    13: "Does your child like climbing on things?",
    14: "Does your child make unusual finger movements near his/her eyes?",
    15: "Is your child interested in other children?",
    16: "Does your child show you things by bringing them to you?",
    17: "When you smile at your child, does he/she smile back at you?",
    18: "Does your child get upset by everyday noises?",
    19: "Does your child walk?",
    20: "Does your child try to copy what you do?",
    21: "Does your child try to get you to watch him/her?",
    22: "Does your child understand when you tell him/her to do something?",
    23: "Does your child look at your face to see how you feel about something new?",
    24: "Does your child like movement activities?",
}

ANSWER_OPTIONS_YN = ["Yes", "No"]

mchat_answers = {}
for i, question in MCHAT_QUESTIONS.items():
    mchat_answers[i] = st.selectbox(
        f"Q{i}: {question}",
        ANSWER_OPTIONS_YN,
        key=f"mchat_{i}",
    )

st.divider()


# ---------------------------------------------------------------------------
# Submit + Prediction
# ---------------------------------------------------------------------------
if st.button("Run Screening", type="primary"):
    payload = {
        "age_mons": int(age_mons),
        "gender": gender,
        "jaundice": jaundice,
        "family_mem_with_asd": family_mem_with_asd,
        "qchat_answers": qchat_answers,
        "mchat_answers": mchat_answers,
    }

    with st.spinner("Running screening model..."):
        result = predict_autism_risk(payload)

    st.success("Screening completed!")

    st.subheader("Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Screening Score", f"{result['screening_score']}/24")
        st.metric("Score Risk Level", result["score_risk_level"])
    with col2:
        st.metric("Default Prediction", result["prediction_default"]["predicted_label"])
        st.metric("Screening Prediction", result["prediction_screening"]["predicted_label"])

    st.write(f"**Referral:** {result['referral_interpretation']}")

    st.write("---")
    st.write("**Class Probabilities:**")
    for label, prob in result["class_probabilities"].items():
        st.progress(prob, text=f"{label}: {prob:.1%}")

    st.info(result["disclaimer"])

    st.session_state["inference_result"] = result


st.divider()


# ---------------------------------------------------------------------------
# Generate Report
# ---------------------------------------------------------------------------
st.subheader("Step 5: Generate Risk Assessment Report")

if st.session_state["inference_result"] is None:
    st.warning("Run screening first to generate a report.")
else:
    if st.button("Generate Report (Groq LLM)"):
        with st.spinner("Generating report using Groq LLM..."):
            report_text = generate_risk_report(st.session_state["inference_result"])

        st.session_state["report_text"] = report_text
        st.success("Report generated successfully!")

    if st.session_state["report_text"] is not None:
        st.text_area("Generated Report", st.session_state["report_text"], height=350)


st.divider()


# ---------------------------------------------------------------------------
# PDF Download
# ---------------------------------------------------------------------------
st.subheader("Step 6: Download PDF")

if st.session_state["report_text"] is None:
    st.warning("Generate the report first to download PDF.")
else:
    if st.button("Generate PDF"):
        with st.spinner("Creating PDF report..."):
            pdf_path = generate_pdf_report(
                st.session_state["inference_result"],
                st.session_state["report_text"],
            )

        st.session_state["pdf_path"] = pdf_path
        st.success("PDF generated!")

    if st.session_state["pdf_path"] is not None:
        pdf_path = st.session_state["pdf_path"]

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name=pdf_path.name,
                mime="application/pdf",
            )
