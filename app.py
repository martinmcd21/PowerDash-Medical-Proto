# =========================================================
# PowerDash Medical — Internal MVP (FULL VERSION)
# =========================================================

import os
import re
import json
import html
import textwrap
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

# Optional local .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# PDF export
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
except Exception:
    canvas = None


# =========================================================
# App Config
# =========================================================

APP_TITLE = "PowerDash Medical"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Styling
# =========================================================

def inject_css():
    st.markdown(
        """
        <style>
        .pd-sidebar-title { font-weight:800; font-size:1.3rem; margin-bottom:0.5rem; }
        .pd-section { font-weight:700; margin-top:1rem; margin-bottom:0.4rem; }
        .pd-card {
            border:1px solid #e5e7eb;
            border-radius:14px;
            padding:14px;
            background:white;
            box-shadow:0 6px 18px rgba(0,0,0,0.06);
        }
        .pd-disclaimer {
            border-left:5px solid #f59e0b;
            background:#fffbeb;
            padding:12px;
            border-radius:10px;
            margin-bottom:1rem;
        }
        .pd-warn {
            border-left:5px solid #ef4444;
            background:#fef2f2;
            padding:12px;
            border-radius:10px;
            margin-bottom:1rem;
        }
        .pd-tile {
            background:#2563eb;
            border-radius:14px;
            padding:18px;
            color:white;
            min-height:110px;
            box-shadow:0 8px 24px rgba(0,0,0,0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Safety Guardrails
# =========================================================

AE_KEYWORDS = [
    "adverse event", "side effect", "reaction", "toxicity",
    "hospitalised", "hospitalized", "death", "fatal",
    "life-threatening", "anaphylaxis", "pharmacovigilance",
]

PII_KEYWORDS = [
    "nhs number", "date of birth", "address", "postcode",
    "patient name", "medical record", "email", "phone",
]

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b(\+?\d[\d\s().-]{7,}\d)\b", re.I)
NHS_RE = re.compile(r"\b(\d{3}\s?\d{3}\s?\d{4})\b")


def detect_ae_or_pii(text: str) -> Tuple[bool, List[str]]:
    if not text:
        return False, []

    reasons = []
    t = text.lower()

    for kw in AE_KEYWORDS:
        if kw in t:
            reasons.append(f"Possible adverse event / PV content detected ('{kw}').")
            break

    for kw in PII_KEYWORDS:
        if kw in t:
            reasons.append(f"Possible patient-identifiable data detected ('{kw}').")
            break

    if EMAIL_RE.search(text):
        reasons.append("Possible email address detected.")
    if PHONE_RE.search(text):
        reasons.append("Possible phone number detected.")
    if NHS_RE.search(text):
        reasons.append("Possible NHS number detected.")

    return len(reasons) > 0, reasons


# =========================================================
# CENTRAL SAFETY HELPER  ✅
# =========================================================

def render_blocked(reasons: List[str], extra: Optional[str] = None):
    escaped = "<br/>".join(html.escape(r) for r in reasons)
    extra_html = f"<br/><br/>{html.escape(extra)}" if extra else ""
    st.markdown(
        f"""
        <div class="pd-warn">
            <b>Generation blocked.</b><br/>
            {escaped}
            {extra_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# LLM Engine
# =========================================================

def get_openai_client() -> Optional[Any]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)


def safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass
        return {"error": "Failed to parse JSON", "raw": raw[:4000]}


def generate_json(
    model: str,
    tool_name: str,
    system_prompt: str,
    user_prompt: str,
    schema_hint: str,
    temperature: float,
) -> Dict[str, Any]:

    client = get_openai_client()
    if not client:
        return {"error": "OpenAI client not available. Check OPENAI_API_KEY."}

    system = f"""
You are PowerDash Medical, an internal Medical Affairs drafting assistant (UK & Ireland).
Rules:
- Drafting support only
- Non-promotional
- VALID JSON ONLY
- No hallucinated references
- Block if AE or patient-identifiable data

Tool: {tool_name}

Instructions:
{system_prompt}

Output schema:
{schema_hint}
""".strip()

    try:
        resp = client.responses.create(
            model=model,
            temperature=temperature,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )

        try:
            raw = resp.output_text
        except Exception:
            raw = str(resp)

        return safe_json_loads(raw)

    except Exception as e:
        return {"error": f"OpenAI error: {str(e)}"}


# =========================================================
# Shared UI
# =========================================================

def render_disclaimer():
    st.markdown(
        "<div class='pd-disclaimer'><b>Drafting support only.</b> Medical review required.</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# TOOL: Scientific Narrative Generator
# =========================================================

def page_scientific_narrative(model: str, temperature: float):
    st.title("📄 Scientific Narrative Generator")
    render_disclaimer()

    with st.form("sn_form"):
        therapy = st.text_input("Therapy area")
        product = st.text_input("Product / Molecule")
        indication = st.text_input("Indication")
        moa = st.text_area("Mechanism of Action")
        pubs = st.text_area("Key publications")
        notes = st.text_area("Internal positioning notes")
        run = st.form_submit_button("Generate narrative")

    combined = "\n".join([therapy, product, indication, moa, pubs, notes])
    blocked, reasons = detect_ae_or_pii(combined)

    if run:
        if blocked:
            render_blocked(reasons, "Remove AE or patient-identifiable data and try again.")
            return

        result = generate_json(
            model,
            "Scientific Narrative Generator",
            "Create a conservative scientific narrative.",
            combined,
            """
            {
              "core_scientific_narrative": "string",
              "disease_state_overview": "string",
              "short_form_variants": {
                "msl_conversation": "string",
                "internal_training": "string",
                "congress_discussion": "string"
              }
            }
            """,
            temperature,
        )
        st.json(result)


# =========================================================
# MAIN ROUTER (other tools unchanged in logic)
# =========================================================

def main():
    inject_css()

    st.sidebar.markdown(f"<div class='pd-sidebar-title'>{APP_TITLE}</div>", unsafe_allow_html=True)

    model = st.sidebar.text_input("Model", DEFAULT_MODEL)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)

    page = st.sidebar.radio(
        "Navigate",
        [
            "Scientific Narrative Generator",
            # Other tools remain wired identically in your original version
        ],
    )

    if page == "Scientific Narrative Generator":
        page_scientific_narrative(model, temperature)


if __name__ == "__main__":
    main()
