# Autism Pre-Screening Tool (Q-CHAT-10 + ML + LLM Report)

This is a Final Year Project (Data Science) implementing an autism pre-screening tool for toddlers using:

- Q-CHAT-10 questionnaire scoring
- Machine Learning classification model
- LLM-based risk assessment report generation (Groq API)
- PDF report export
- Streamlit UI

## Features
- Parent-friendly Q-CHAT-10 form
- Automatic Q-CHAT scoring + risk level
- ML probability prediction for ASD traits
- LLM-generated screening report
- Downloadable PDF report

## Tech Stack
- Python
- scikit-learn
- XGBoost
- Streamlit
- Groq API (Llama 3.1)
- ReportLab (PDF)

## Setup Instructions

### 1. Create virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run the FastAPI app
```bash
python -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

The FastAPI server now serves the web frontend from `web/public`, and that
frontend calls the `/api/...` endpoints exposed by `server/app.py`.

### Optional: Run the Streamlit app instead
```bash
streamlit run app/ui/streamlit_app.py
```
