import os
import re
import json
import textwrap
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

# Optional: load .env locally (no persistence; just reads local environment)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenAI (Python SDK v1+)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# PDF export (reportlab)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
except Exception:
    canvas = None


# =========================
# App Config
# =========================

APP_TITLE = "PowerDash Medical"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # sensible default; override via env var
UKI_DEFAULT = "United Kingdom"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Styling (PowerDash-inspired)
# =========================

def inject_css() -> None:
    st.markdown(
        """
        <style>
          .pd-sidebar-title {
              font-weight: 800;
              font-size: 1.25rem;
              margin: 0.25rem 0 0.75rem 0;
          }
          .pd-section {
              font-weight: 700;
              font-size: 0.95rem;
              margin-top: 1rem;
              margin-bottom: 0.35rem;
              color: #111827;
          }
          .pd-tile {
              background: #2563eb;
              border-radius: 14px;
              padding: 18px 16px;
              color: white;
              min-height: 110px;
              display: flex;
              flex-direction: column;
              justify-content: space-between;
              box-shadow: 0 8px 24px rgba(0,0,0,0.12);
              border: 1px solid rgba(255,255,255,0.15);
          }
          .pd-tile h3 {
              margin: 0;
              padding: 0;
              font-size: 1.02rem;
              line-height: 1.25rem;
              color: white;
          }
          .pd-tile p {
              margin: 0.35rem 0 0 0;
              opacity: 0.9;
              font-size: 0.88rem;
          }
          .pd-small-muted {
              color: #6b7280;
              font-size: 0.9rem;
          }
          .pd-disclaimer {
              border-left: 5px solid #f59e0b;
              background: #fffbeb;
              padding: 12px 14px;
              border-radius: 10px;
              color: #111827;
              margin: 0.5rem 0 1rem 0;
          }
          .pd-warn {
              border-left: 5px solid #ef4444;
              background: #fef2f2;
              padding: 12px 14px;
              border-radius: 10px;
              color: #111827;
              margin: 0.5rem 0 1rem 0;
          }
          .pd-card {
              border: 1px solid #e5e7eb;
              border-radius: 14px;
              padding: 14px 14px;
              background: white;
              box-shadow: 0 6px 18px rgba(0,0,0,0.06);
          }
          .pd-hr {
              margin: 0.5rem 0 1rem 0;
          }
          .pd-json {
              font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
              font-size: 0.9rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Global Guardrails (AE + Patient Identifiable)
# =========================

AE_KEYWORDS = [
    "adverse event", "ae", "side effect", "reaction", "overdose", "toxicity",
    "hospitalised", "hospitalized", "death", "died", "fatal", "life-threatening",
    "anaphylaxis", "serious adverse", "suspected adverse", "pharmacovigilance",
    "yellow card", "mhra report", "report an event", "drug safety report",
]

# Very simple PII heuristics (NOT exhaustive)
PII_KEYWORDS = [
    "nhs number", "date of birth", "dob", "address", "postcode", "patient name",
    "telephone", "phone", "email", "medical record", "mrn", "national insurance",
]

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"\b(\+?\d[\d\s().-]{7,}\d)\b", re.I)
# NHS number: 10 digits, usually spaced 3-3-4; this is a heuristic
NHS_RE = re.compile(r"\b(\d{3}\s?\d{3}\s?\d{4})\b")


def detect_ae_or_pii(text: str) -> Tuple[bool, List[str]]:
    """
    Returns (blocked, reasons).
    Conservative: if AE or patient-identifiable content is suspected, block generation.
    """
    if not text or not text.strip():
        return (False, [])

    t = text.lower()
    reasons = []

    # AE keyword hit
    for kw in AE_KEYWORDS:
        if kw in t:
            reasons.append(f"Possible adverse event / PV content detected (keyword: '{kw}').")
            break

    # PII keyword hit
    for kw in PII_KEYWORDS:
        if kw in t:
            reasons.append(f"Possible patient-identifiable data detected (keyword: '{kw}').")
            break

    # Regex indicators
    if EMAIL_RE.search(text):
        reasons.append("Possible email address detected.")
    if NHS_RE.search(text):
        reasons.append("Possible NHS number detected.")
    # Phone is noisy; still block if found (internal conservative posture)
    if PHONE_RE.search(text):
        reasons.append("Possible phone number detected.")

    blocked = len(reasons) > 0
    return blocked, reasons


# =========================
# Shared LLM Engine
# =========================

def get_openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def safe_json_loads(raw: str) -> Dict[str, Any]:
    """
    Try to parse JSON robustly. If it fails, return a structured fallback.
    """
    try:
        return json.loads(raw)
    except Exception:
        # Attempt to recover JSON substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start:end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
        return {
            "error": "Failed to parse JSON output from model.",
            "raw_output": raw[:8000],
        }


def generate_json(
    model: str,
    tool_name: str,
    system_prompt: str,
    user_prompt: str,
    json_schema_hint: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Shared generation helper (one engine used by all tools).
    - Uses strict "Return JSON only" instruction.
    - Does NOT allow references beyond user input (enforced by prompt + tool guardrails).
    """
    client = get_openai_client()
    if client is None:
        return {"error": "OpenAI client not available. Ensure OPENAI_API_KEY is set and openai package is installed."}

    # Strong JSON-only constraint
    sys = f"""
You are PowerDash Medical, an internal Medical Affairs drafting assistant (UK & Ireland default).
Conservative tone. Non-promotional intent. Drafting support only; medical review required.

ABSOLUTE RULES:
- Return VALID JSON ONLY. No markdown. No extra commentary.
- If asked for references: ONLY use citations that appear in the user-provided input. Never invent references.
- Do not produce promotional claims. Avoid superlatives. Use balanced, compliant language.
- If content implies adverse events or includes patient-identifiable information: do NOT proceed. Return JSON with:
  {{ "blocked": true, "reason": "...", "safe_next_step": "..." }}

Tool context: {tool_name}

Tool-specific system instructions:
{system_prompt}

Output JSON schema (follow this structure exactly):
{json_schema_hint}
""".strip()

    usr = f"""
USER INPUT (treat as confidential and session-only; do not store):
{user_prompt}
""".strip()

    try:
        # Using Responses API style via SDK v1: client.responses.create(...)
        # This is compatible with most current OpenAI SDKs.
        resp = client.responses.create(
            model=model,
            temperature=temperature,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr},
            ],
        )
        # The SDK returns output text in different shapes depending on version.
        # We'll robustly pull text.
        raw_text = ""
        try:
            raw_text = resp.output_text  # preferred
        except Exception:
            # fallback: iterate output
            try:
                parts = []
                for item in resp.output:
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            parts.append(c.get("text", ""))
                raw_text = "\n".join(parts)
            except Exception:
                raw_text = str(resp)

        return safe_json_loads(raw_text)

    except Exception as e:
        return {"error": f"OpenAI API call failed: {type(e).__name__}: {str(e)}"}


# =========================
# Utility: Copy + PDF Export
# =========================

def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="pd-disclaimer">
          <b>Drafting support only.</b> Outputs are for internal drafting support and require medical, legal,
          and regulatory review before use. Non-promotional intent. UK & Ireland default tone.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_copy_button(text: str, key: str) -> None:
    """
    Streamlit doesn't have a native clipboard API. This uses a tiny JS snippet.
    Note: Works in most modern browsers; some locked-down environments may block clipboard access.
    """
    # Escape safely for JS string
    js_text = json.dumps(text)

    html = f"""
    <div style="display:flex; gap: 8px; align-items:center; margin: 6px 0 10px 0;">
      <button
        style="
          background:#111827;color:white;border:none;border-radius:10px;
          padding:8px 12px;cursor:pointer;font-weight:700;
        "
        onclick="navigator.clipboard.writeText({js_text}).then(() => {{
          const el = document.getElementById('{key}_status');
          if (el) el.innerText = 'Copied ✓';
          setTimeout(() => {{ if (el) el.innerText = ''; }}, 1500);
        }})"
      >
        Copy to clipboard
      </button>
      <span id="{key}_status" style="color:#059669;font-weight:700;"></span>
    </div>
    """
    st.components.v1.html(html, height=55)


def build_pdf_bytes(title: str, sections: List[Tuple[str, str]]) -> Optional[bytes]:
    """
    Build a simple PDF (A4) from titled sections.
    Uses reportlab if installed; otherwise returns None.
    """
    if canvas is None:
        return None

    from io import BytesIO
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Margins
    left = 18 * mm
    right = 18 * mm
    top = height - 18 * mm
    bottom = 18 * mm
    y = top

    def draw_wrapped(text: str, y_pos: float, font="Helvetica", size=10, leading=13) -> float:
        c.setFont(font, size)
        max_width = width - left - right
        # Approximate characters per line; wrap using textwrap
        # (simple approach; good enough for MVP)
        chars_per_line = max(60, int(max_width / (size * 0.55)))
        lines = []
        for paragraph in text.split("\n"):
            if paragraph.strip() == "":
                lines.append("")
            else:
                lines.extend(textwrap.wrap(paragraph, width=chars_per_line))
        for line in lines:
            if y_pos < bottom + 30:
                c.showPage()
                y_pos = top
                c.setFont(font, size)
            c.drawString(left, y_pos, line)
            y_pos -= leading
        return y_pos

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, title)
    y -= 18

    c.setFont("Helvetica", 9)
    c.drawString(left, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 18

    for sec_title, sec_body in sections:
        if y < bottom + 60:
            c.showPage()
            y = top

        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, sec_title)
        y -= 14
        y = draw_wrapped(sec_body.strip(), y, font="Helvetica", size=10, leading=13)
        y -= 10

    c.save()
    buffer.seek(0)
    return buffer.read()


def sections_from_json(payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Turn a JSON dict into (section_title, section_text) pairs for display/PDF.
    Keeps stable ordering where possible.
    """
    sections = []
    for k, v in payload.items():
        if isinstance(v, (dict, list)):
            sec_text = json.dumps(v, indent=2, ensure_ascii=False)
        else:
            sec_text = "" if v is None else str(v)
        sections.append((k.replace("_", " ").title(), sec_text))
    return sections


def render_output_block(tool_title: str, result: Dict[str, Any], key_prefix: str) -> None:
    """
    Standard output rendering:
    - JSON viewer
    - "pretty text" (best-effort)
    - Copy + PDF download
    """
    if not result:
        return

    # Blocked handling
    if result.get("blocked") is True:
        st.markdown(
            f"""
            <div class="pd-warn">
              <b>Generation blocked.</b><br/>
              Reason: {st._utils.escape_markdown(str(result.get("reason", "Safety trigger.")))}<br/>
              Next step: {st._utils.escape_markdown(str(result.get("safe_next_step", "Remove AE/PII and try again.")))}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.json(result)
        return

    if "error" in result:
        st.error(result["error"])
        st.json(result)
        return

    tabs = st.tabs(["Formatted output", "JSON"])
    with tabs[1]:
        st.json(result)

    # Best-effort: create a readable text bundle for clipboard/PDF
    text_lines = [f"{tool_title}", "-" * len(tool_title)]
    for sec_title, sec_body in sections_from_json(result):
        text_lines.append(f"\n{sec_title}\n" + ("~" * len(sec_title)))
        text_lines.append(sec_body)
    full_text = "\n".join(text_lines).strip()

    with tabs[0]:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        render_copy_button(full_text, key=f"{key_prefix}_copy")
        st.text_area("Output (editable for your internal drafting)", value=full_text, height=380, key=f"{key_prefix}_text")
        st.download_button(
            "Download as .txt",
            data=full_text.encode("utf-8"),
            file_name=f"powerdash_medical_{key_prefix}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        pdf_bytes = build_pdf_bytes(tool_title, sections_from_json(result))
        if pdf_bytes is None:
            st.info("PDF export unavailable (reportlab not installed). Add reportlab to requirements.txt to enable.")
        else:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"powerdash_medical_{key_prefix}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Prompts + Schemas per Tool
# =========================

TOOL_DEFS = {
    "Home": {
        "icon": "🏠",
        "group": "Home",
        "desc": "Tool tiles and quick access.",
    },

    # Core
    "Scientific Narrative Generator": {
        "icon": "📄",
        "group": "Core Medical Affairs Tools",
        "desc": "Core narrative + disease state + short-form variants.",
        "system_prompt": """
Write a conservative scientific narrative suitable for internal Medical Affairs drafting.
Use only the content provided by the user. If the user provides publications, you may cite them by name exactly as provided.
Avoid definitive efficacy/safety claims. Use balanced language (e.g., 'may', 'evidence suggests').
""".strip(),
        "schema": """
{
  "blocked": false,
  "core_scientific_narrative": "string",
  "disease_state_overview": "string",
  "short_form_variants": {
    "msl_conversation": "string",
    "internal_training": "string",
    "congress_discussion": "string"
  },
  "assumptions_and_gaps": ["string"],
  "suggested_next_inputs": ["string"]
}
""".strip(),
    },

    "MSL Briefing Pack Generator": {
        "icon": "🧠",
        "group": "Core Medical Affairs Tools",
        "desc": "Objectives, messages, discussion guide, Q&A, do/don’t.",
        "system_prompt": """
Create an MSL briefing pack in a UK & Ireland-appropriate tone.
Non-promotional. Focus on scientific exchange, needs-based engagement, and compliant language.
Do not invent evidence. If evidence is missing, clearly label gaps and propose questions to validate.
""".strip(),
        "schema": """
{
  "blocked": false,
  "briefing_pack": {
    "purpose_and_context": "string",
    "objectives": ["string"],
    "key_scientific_messages": ["string"],
    "stakeholder_hypotheses": ["string"],
    "discussion_guide": {
      "opening": ["string"],
      "core_questions": ["string"],
      "closing": ["string"]
    },
    "anticipated_questions_and_draft_answers": [
      {"question": "string", "draft_answer": "string", "confidence": "low|medium|high", "evidence_basis": "string"}
    ],
    "do_and_dont": {
      "do": ["string"],
      "dont": ["string"]
    },
    "follow_up_actions": ["string"]
  },
  "assumptions_and_gaps": ["string"]
}
""".strip(),
    },

    "Medical Information Response Generator": {
        "icon": "📚",
        "group": "Core Medical Affairs Tools",
        "desc": "MI response with guardrails; references only from user input.",
        "system_prompt": """
Draft a Medical Information (MI) response suitable for UK/Ireland.
Strict rules:
- No promotional language.
- No hallucinated references.
- Reference list MUST be ONLY items provided by the user (verbatim titles/identifiers).
- If insufficient evidence is provided, state limitations and request additional sources.
Also: if the user input includes potential AE or patient-identifiable data, return blocked JSON.
""".strip(),
        "schema": """
{
  "blocked": false,
  "country": "UK|Ireland",
  "audience": "HCP|Pharmacist|Payer",
  "long_form_written_response": "string",
  "short_verbal_response": "string",
  "reference_list": ["string"],
  "limitations_and_required_follow_up": ["string"]
}
""".strip(),
    },

    "Congress & Advisory Board Planner": {
        "icon": "🎤",
        "group": "Core Medical Affairs Tools",
        "desc": "Agenda, discussion guide, question bank, insight capture.",
        "system_prompt": """
Plan a compliant Congress or Advisory Board engagement.
Keep tone conservative. Avoid promotional framing. Focus on scientific exchange and insight capture.
Do not invent clinical claims. If details are missing, propose options and note assumptions.
""".strip(),
        "schema": """
{
  "blocked": false,
  "event_type": "Congress|Advisory Board",
  "agenda": [
    {"timebox": "string", "session_title": "string", "purpose": "string", "facilitation_notes": "string"}
  ],
  "discussion_guide": {
    "opening": ["string"],
    "topics_and_prompts": ["string"],
    "closing": ["string"]
  },
  "question_bank": ["string"],
  "insight_capture_framework": {
    "capture_fields": ["string"],
    "rating_scales": ["string"],
    "debrief_structure": ["string"]
  },
  "assumptions_and_gaps": ["string"]
}
""".strip(),
    },

    # Additional
    "Insight Capture & Thematic Analysis": {
        "icon": "📊",
        "group": "Additional Tools",
        "desc": "In-session insights → themes, signals, exec summary.",
        "system_prompt": """
You are analysing qualitative Medical Affairs insights.
Group into themes, distinguish signal vs noise, and produce an executive-ready summary.
Do not add external facts. Only work from provided insights.
""".strip(),
        "schema": """
{
  "blocked": false,
  "thematic_grouping": [
    {"theme": "string", "summary": "string", "supporting_insights": ["string"]}
  ],
  "emerging_themes": ["string"],
  "signal_vs_noise_summary": {
    "signals": ["string"],
    "noise_or_low_confidence": ["string"],
    "watch_list": ["string"]
  },
  "executive_ready_summary": "string"
}
""".strip(),
    },

    "Medical Affairs Executive Summary Generator": {
        "icon": "📈",
        "group": "Additional Tools",
        "desc": "Leadership-ready summary with risks and opportunities.",
        "system_prompt": """
Create a leadership-ready Medical Affairs executive summary from provided pasted outputs.
Conservative, clear, non-promotional. Identify themes, risks, opportunities, and recommended next steps.
Do not invent data; base only on provided text.
""".strip(),
        "schema": """
{
  "blocked": false,
  "executive_summary": "string",
  "top_themes": ["string"],
  "risks": ["string"],
  "opportunities": ["string"],
  "recommended_next_steps": ["string"],
  "open_questions": ["string"]
}
""".strip(),
    },

    "Compliance & Governance Summary": {
        "icon": "🔒",
        "group": "Additional Tools",
        "desc": "Static governance page (no generation).",
    },

    "Medical Affairs SOP Drafting Tool": {
        "icon": "📑",
        "group": "Additional Tools",
        "desc": "Draft SOP in conservative regulatory tone.",
        "system_prompt": """
Draft a Medical Affairs SOP in a conservative regulatory tone suitable for UK/Ireland default.
Structure: Purpose, Scope, Definitions, Roles & Responsibilities, Procedure, Documentation, Training, Compliance, Version Control.
If existing SOP text is provided, adapt and improve it; do not copy promotional language.
Do not invent company-specific systems; label placeholders clearly.
""".strip(),
        "schema": """
{
  "blocked": false,
  "sop_title": "string",
  "sop_version": "string",
  "purpose": "string",
  "scope": "string",
  "definitions": ["string"],
  "roles_and_responsibilities": ["string"],
  "procedure_steps": ["string"],
  "documentation_and_records": ["string"],
  "training_requirements": ["string"],
  "compliance_and_governance": ["string"],
  "version_control": ["string"],
  "assumptions_and_gaps": ["string"]
}
""".strip(),
    },
}


# =========================
# Sidebar Navigation
# =========================

def sidebar_nav() -> Tuple[str, str, float]:
    """
    Returns (selected_page, model, temperature).
    """
    st.sidebar.markdown(f"<div class='pd-sidebar-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.sidebar.caption("Internal Medical Affairs AI workbench (stateless, drafting support only).")

    # Model controls (internal MVP)
    with st.sidebar.expander("LLM settings (internal)", expanded=False):
        model = st.text_input("Model", value=DEFAULT_MODEL, help="Override via OPENAI_MODEL env var.")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        st.caption("Lower temperature is recommended for compliant drafting.")

    st.sidebar.markdown("<div class='pd-section'>Home</div>", unsafe_allow_html=True)
    home_choice = st.sidebar.radio(
        "Navigate",
        options=["Home"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div class='pd-section'>Core Medical Affairs Tools</div>", unsafe_allow_html=True)
    core_choice = st.sidebar.radio(
        "Core tools",
        options=[
            "Scientific Narrative Generator",
            "MSL Briefing Pack Generator",
            "Medical Information Response Generator",
            "Congress & Advisory Board Planner",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div class='pd-section'>Additional Tools</div>", unsafe_allow_html=True)
    add_choice = st.sidebar.radio(
        "Additional tools",
        options=[
            "Insight Capture & Thematic Analysis",
            "Medical Affairs Executive Summary Generator",
            "Compliance & Governance Summary",
            "Medical Affairs SOP Drafting Tool",
        ],
        label_visibility="collapsed",
    )

    # Pick whichever the user clicked last by using session_state
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Home"

    # Update selection when a radio changes (simple heuristic)
    # Streamlit reruns; we can set based on which changed most recently by checking widget values.
    # For MVP, treat core/add radios as primary if not Home.
    if home_choice == "Home" and st.session_state.selected_page == "Home":
        selected = "Home"
    else:
        # Use currently selected in core/add to override Home
        # If user wants Home, they can click a tile or refresh selection.
        selected = st.session_state.get("selected_page", "Home")

    # If user clicks on any radio, update selected_page accordingly:
    # We always update in a stable order: if a core tool selected, that becomes page; otherwise additional; otherwise home.
    # This feels natural in use.
    if core_choice:
        selected = core_choice
    if add_choice:
        # If the user clicked an additional tool, it becomes selected
        # (This will be true every run; so we need a tie-breaker)
        # We'll check last_clicked_group stored via tile clicks. For radio-only MVP, treat the last changed as selected:
        # Can't reliably detect "changed" without callbacks, so we bias toward core unless user is on additional page already.
        if st.session_state.get("selected_page_group") == "Additional":
            selected = add_choice

    # Set selected_page_group based on selection
    if selected in ["Scientific Narrative Generator", "MSL Briefing Pack Generator", "Medical Information Response Generator", "Congress & Advisory Board Planner"]:
        st.session_state.selected_page_group = "Core"
    elif selected in ["Insight Capture & Thematic Analysis", "Medical Affairs Executive Summary Generator", "Compliance & Governance Summary", "Medical Affairs SOP Drafting Tool"]:
        st.session_state.selected_page_group = "Additional"
    else:
        st.session_state.selected_page_group = "Home"

    st.session_state.selected_page = selected

    return selected, model, float(temperature)


def tile_button(label: str, description: str, page_name: str, icon: str) -> None:
    """
    A PowerDash-style tile. Uses a button below the tile to ensure accessibility.
    """
    st.markdown(
        f"""
        <div class="pd-tile">
          <div>
            <h3>{icon} {label}</h3>
            <p>{description}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(f"Open: {label}", key=f"tile_{page_name}", use_container_width=True):
        st.session_state.selected_page = page_name
        if TOOL_DEFS[page_name]["group"] == "Additional Tools":
            st.session_state.selected_page_group = "Additional"
        elif TOOL_DEFS[page_name]["group"] == "Core Medical Affairs Tools":
            st.session_state.selected_page_group = "Core"
        else:
            st.session_state.selected_page_group = "Home"
        st.rerun()


# =========================
# Pages
# =========================

def page_home() -> None:
    st.title("PowerDash Medical")
    render_disclaimer()

    st.markdown(
        """
        <div class="pd-card">
          <b>Internal MVP:</b> A multi-tool Medical Affairs drafting workbench inspired by PowerDash HR.<br/>
          <span class="pd-small-muted">Stateless • No data retention beyond session • No authentication • Streamlit-only</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='pd-hr'/>", unsafe_allow_html=True)

    st.subheader("Core Medical Affairs Tools")
    c1, c2, c3, c4 = st.columns(4, gap="large")
    core_tools = [
        "Scientific Narrative Generator",
        "MSL Briefing Pack Generator",
        "Medical Information Response Generator",
        "Congress & Advisory Board Planner",
    ]
    for col, name in zip([c1, c2, c3, c4], core_tools):
        with col:
            d = TOOL_DEFS[name]
            tile_button(name, d["desc"], name, d["icon"])

    st.subheader("Additional Tools")
    a1, a2, a3, a4 = st.columns(4, gap="large")
    additional_tools = [
        "Insight Capture & Thematic Analysis",
        "Medical Affairs Executive Summary Generator",
        "Compliance & Governance Summary",
        "Medical Affairs SOP Drafting Tool",
    ]
    for col, name in zip([a1, a2, a3, a4], additional_tools):
        with col:
            d = TOOL_DEFS[name]
            tile_button(name, d["desc"], name, d["icon"])


def page_scientific_narrative(model: str, temperature: float) -> None:
    st.title("📄 Scientific Narrative Generator")
    render_disclaimer()

    with st.form("sn_form"):
        therapy_area = st.text_input("Therapy area", placeholder="e.g., Oncology, Dermatology, Respiratory")
        product = st.text_input("Product / Molecule", placeholder="e.g., Molecule X / Brand Y")
        indication = st.text_input("Indication", placeholder="e.g., Moderate-to-severe condition Z")
        moa = st.text_area("Mechanism of Action", height=110, placeholder="Describe MOA in scientific terms.")
        pubs = st.text_area("Key publications (paste titles / abstracts / key excerpts)", height=140)
        positioning = st.text_area("Internal positioning notes (non-promotional, factual)", height=140)

        run = st.form_submit_button("Generate narrative suite", use_container_width=True)

    combined_for_safety = "\n".join([therapy_area, product, indication, moa, pubs, positioning])
    blocked, reasons = detect_ae_or_pii(combined_for_safety)

    if run:
        if blocked:
            st.markdown(
                "<div class='pd-warn'><b>Generation blocked.</b><br/>" +
                "<br/>".join([st._utils.escape_markdown(r) for r in reasons]) +
                "<br/><br/>Remove AE / patient-identifiable data and try again.</div>",
                unsafe_allow_html=True,
            )
            return

        user_prompt = f"""
Therapy area: {therapy_area}
Product/Molecule: {product}
Indication: {indication}
Mechanism of Action: {moa}

Key publications (user provided):
{pubs}

Internal positioning notes (user provided):
{positioning}
""".strip()

        result = generate_json(
            model=model,
            tool_name="Scientific Narrative Generator",
            system_prompt=TOOL_DEFS["Scientific Narrative Generator"]["system_prompt"],
            user_prompt=user_prompt,
            json_schema_hint=TOOL_DEFS["Scientific Narrative Generator"]["schema"],
            temperature=temperature,
        )
        render_output_block("Scientific Narrative Suite", result, key_prefix="scientific_narrative")


def page_msl_briefing_pack(model: str, temperature: float) -> None:
    st.title("🧠 MSL Briefing Pack Generator")
    render_disclaimer()

    with st.form("msl_form"):
        therapy_area = st.text_input("Therapy area", placeholder="e.g., Immunology")
        product = st.text_input("Product / Molecule", placeholder="e.g., Molecule X")
        indication = st.text_input("Indication / Use context", placeholder="e.g., Condition Y in adults")
        stakeholder = st.selectbox("Stakeholder type", ["KOL", "HCP", "Pharmacist", "Payer/HTA", "Other"])
        engagement_goal = st.text_area("Engagement objectives (what do you need to learn/achieve?)", height=120)
        key_evidence = st.text_area("Evidence provided by user (paste excerpts / summaries / publications)", height=160)
        constraints = st.text_area("Constraints / considerations (geography, setting, compliance notes)", height=110, value="UK & Ireland default. Non-promotional. Scientific exchange only.")

        run = st.form_submit_button("Generate briefing pack", use_container_width=True)

    combined_for_safety = "\n".join([therapy_area, product, indication, stakeholder, engagement_goal, key_evidence, constraints])
    blocked, reasons = detect_ae_or_pii(combined_for_safety)

    if run:
        if blocked:
            st.markdown(
                "<div class='pd-warn'><b>Generation blocked.</b><br/>" +
                "<br/>".join([st._utils.escape_markdown(r) for r in reasons]) +
                "<br/><br/>Remove AE / patient-identifiable data and try again.</div>",
                unsafe_allow_html=True,
            )
            return

        user_prompt = f"""
Therapy area: {therapy_area}
Product/Molecule: {product}
Indication/context: {indication}
Stakeholder type: {stakeholder}

Engagement objectives:
{engagement_goal}

Evidence provided by user:
{key_evidence}

Constraints/considerations:
{constraints}
""".strip()

        result = generate_json(
            model=model,
            tool_name="MSL Briefing Pack Generator",
            system_prompt=TOOL_DEFS["MSL Briefing Pack Generator"]["system_prompt"],
            user_prompt=user_prompt,
            json_schema_hint=TOOL_DEFS["MSL Briefing Pack Generator"]["schema"],
            temperature=temperature,
        )
        render_output_block("MSL Briefing Pack", result, key_prefix="msl_briefing")


def page_mi_response(model: str, temperature: float) -> None:
    st.title("📚 Medical Information Response Generator")
    render_disclaimer()

    st.info("Guardrails: blocks generation if suspected AE/PV content or patient-identifiable data is detected.")

    with st.form("mi_form"):
        country = st.selectbox("Country", ["UK", "Ireland"])
        audience = st.selectbox("Audience", ["HCP", "Pharmacist", "Payer"])
        question = st.text_area("Medical question", height=120, placeholder="Enter the medical question exactly as received (no patient identifiers).")
        evidence = st.text_area("Evidence provided by user (paste excerpts, citations, or internal text)", height=180, help="References MUST be included here if you want them cited.")

        run = st.form_submit_button("Generate MI response", use_container_width=True)

    combined_for_safety = "\n".join([country, audience, question, evidence])
    blocked, reasons = detect_ae_or_pii(combined_for_safety)

    if run:
        if blocked:
            st.markdown(
                "<div class='pd-warn'><b>Generation blocked.</b><br/>" +
                "<br/>".join([st._utils.escape_markdown(r) for r in reasons]) +
                "<br/><br/>This tool cannot process adverse events (PV) or patient-identifiable data. Remove it and try again.</div>",
                unsafe_allow_html=True,
            )
            return

        user_prompt = f"""
Country: {country}
Audience: {audience}
Medical question:
{question}

Evidence provided by user (ONLY source for references):
{evidence}
""".strip()

        result = generate_json(
            model=model,
            tool_name="Medical Information Response Generator",
            system_prompt=TOOL_DEFS["Medical Information Response Generator"]["system_prompt"],
            user_prompt=user_prompt,
            json_schema_hint=TOOL_DEFS["Medical Information Response Generator"]["schema"],
            temperature=temperature,
        )
        render_output_block("Medical Information Response", result, key_prefix="mi_response")


def page_congress_adboard(model: str, temperature: float) -> None:
    st.title("🎤 Congress & Advisory Board Planner")
    render_disclaimer()

    with st.form("cab_form"):
        event_type = st.selectbox("Event type", ["Congress", "Advisory Board"])
        therapy_area = st.text_input("Therapy area", placeholder="e.g., Cardiometabolic")
        audience_type = st.text_input("Audience type", placeholder="e.g., MSL team, KOLs, MDT, pharmacists")
        objectives = st.text_area("Objectives", height=120, placeholder="What do you need to learn, align, or explore?")
        geography = st.selectbox("Geography", ["UK", "Ireland", "UK & Ireland", "Other"])
        constraints = st.text_area("Constraints / notes (optional)", height=100, value="Non-promotional. Scientific exchange. Ensure appropriate compliance review.")

        run = st.form_submit_button("Generate plan", use_container_width=True)

    combined_for_safety = "\n".join([event_type, therapy_area, audience_type, objectives, geography, constraints])
    blocked, reasons = detect_ae_or_pii(combined_for_safety)

    if run:
        if blocked:
            st.markdown(
                "<div class='pd-warn'><b>Generation blocked.</b><br/>" +
                "<br/>".join([st._utils.escape_markdown(r) for r in reasons]) +
                "<br/><br/>Remove AE / patient-identifiable data and try again.</div>",
                unsafe_allow_html=True,
            )
            return

        user_prompt = f"""
Event type: {event_type}
Therapy area: {therapy_area}
Audience type: {audience_type}
Objectives:
{objectives}
Geography: {geography}
Constraints/notes:
{constraints}
""".strip()

        result = generate_json(
            model=model,
            tool_name="Congress & Advisory Board Planner",
            system_prompt=TOOL_DEFS["Congress & Advisory Board Planner"]["system_prompt"],
            user_prompt=user_prompt,
            json_schema_hint=TOOL_DEFS["Congress & Advisory Board Planner"]["schema"],
            temperature=temperature,
        )
        render_output_block(f"{event_type} Plan", result, key_prefix="event_planner")


def page_insight_analysis(model: str, temperature: float) -> None:
    st.title("📊 Insight Capture & Thematic Analysis")
    render_disclaimer()

    # Session-only collection (cleared on refresh/new session)
    if "insights" not in st.session_state:
        st.session_state.insights = []

    with st.form("insight_add"):
        col1, col2 = st.columns(2)
        with col1:
            region = st.text_input("Region", value="UK & Ireland")
        with col2:
            stakeholder_type = st.selectbox("Stakeholder type", ["KOL", "HCP", "Pharmacist", "Payer/HTA", "Other"])

        insight_text = st.text_area("Insight (free text)", height=120, placeholder="Enter one insight per submission (no patient identifiers).")
        add = st.form_submit_button("Add insight to session list", use_container_width=True)

    if add:
        combined = "\n".join([region, stakeholder_type, insight_text])
        blocked, reasons = detect_ae_or_pii(combined)
        if blocked:
            st.markdown(
                "<div class='pd-warn'><b>Cannot add insight.</b><br/>" +
                "<br/>".join([st._utils.escape_markdown(r) for r in reasons]) +
                "<br/><br/>Remove AE / patient-identifiable data and try again.</div>",
                unsafe_allow_html=True,
            )
        elif insight_text.strip():
            st.session_state.insights.append({
                "region": region.strip(),
                "stakeholder_type": stakeholder_type,
                "insight": insight_text.strip(),
            })
            st.success("Insight added (session-only).")

    st.markdown("<hr class='pd-hr'/>", unsafe_allow_html=True)

    st.subheader("Session insights (not saved)")
    if not st.session_state.insights:
        st.caption("No insights yet. Add a few above, then run analysis.")
    else:
        for i, item in enumerate(st.session_state.insights, start=1):
            st.markdown(
                f"<div class='pd-card'><b>{i}. {item['stakeholder_type']} • {item['region']}</b><br/>{st._utils.escape_markdown(item['insight'])}</div>",
                unsafe_allow_html=True,
            )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Clear session insights", use_container_width=True):
            st.session_state.insights = []
            st.rerun()

    with c2:
        run = st.button("Run thematic analysis", use_container_width=True)

    if run:
        # Safety check again (conservative)
        combined = "\n".join([x["insight"] for x in st.session_state.insights])
        blocked, reasons = detect_ae_or_pii(combined)
        if blocked:
            st.markdown(
                "<div class='pd-warn'><b>Analysis blocked.</b><br/>" +
                "<br/>".join([st._utils.escape_markdown(r) for r in reasons]) +
                "<br/><br/>Remove AE / patient-identifiable data from insights and try again.</div>",
                unsafe_allow_html=True,
            )
            return

        user_prompt = "Insights (session-only):\n" + json.dumps(st.session_state.insights, indent=2, ensure_ascii=False)

        result = generate_json(
            model=model,
            tool_name="Insight Capture & Thematic Analysis",
            system_prompt=TOOL_DEFS["Insight Capture & Thematic Analysis"]["system_prompt"],
            user_prompt=user_prompt,
            json_schema_hint=TOOL_DEFS["Insight Capture & Thematic Analysis"]["schema"],
            temperature=temperature,
        )
        render_output_block("Insight Thematic Analysis", result, key_prefix="insight_analysis")


def page_exec_summary(model: str, temperature: float) -> None:
    st.title("📈 Medical Affairs Executive Summary Generator")
    render_disclaimer()

    with st.form("execsum_form"):
        pasted_outputs = st.text_area(
            "Paste selected outputs from other tools (session-only)",
            height=220,
            placeholder="Paste narrative, MI response, event plan, insight analysis, etc."
        )
        strategic_focus = st.text_input("Strategic focus", placeholder="e.g., H2 priorities, KOL strategy, evidence-generation, congress planning")

        run = st.form_submit_button("Generate executive summary", use_container_width=True)

    blocked, reasons = detect_ae_or_pii("\n".join([pasted_outputs, strategic_focus]))

    if run:
        if blocked:
            st.markdown(
                "<div class='pd-warn'><b>Generation blocked.</b><br/>" +
                "<br/>".join([st._utils.escape_markdown(r) for r in reasons]) +
                "<br/><br/>Remove AE / patient-identifiable data and try again.</div>",
                unsafe_allow_html=True,
            )
            return

        user_prompt = f"""
Strategic focus: {strategic_focus}

Pasted tool outputs (user provided):
{pasted_outputs}
""".strip()

        result = generate_json(
            model=model,
            tool_name="Medical Affairs Executive Summary Generator",
            system_prompt=TOOL_DEFS["Medical Affairs Executive Summary Generator"]["system_prompt"],
            user_prompt=user_prompt,
            json_schema_hint=TOOL_DEFS["Medical Affairs Executive Summary Generator"]["schema"],
            temperature=temperature,
        )
        render_output_block("Medical Affairs Executive Summary", result, key_prefix="exec_summary")


def page_compliance() -> None:
    st.title("🔒 Compliance & Governance Summary")
    st.markdown(
        """
        <div class="pd-card">
          <h3 style="margin-top:0;">PowerDash Medical — Internal Governance</h3>
          <ul>
            <li><b>Stateless:</b> No database. No storage of user content. Inputs/outputs exist only in-session.</li>
            <li><b>No authentication:</b> Intended for controlled internal environments only (MVP foundation).</li>
            <li><b>No analytics:</b> No tracking of usage or user behaviour in this MVP.</li>
            <li><b>Drafting support only:</b> Outputs require medical, legal, and regulatory review before use.</li>
            <li><b>Non-promotional intent:</b> Tools are designed for scientific exchange and internal drafting.</li>
            <li><b>References:</b> For MI responses, reference lists must come <u>only</u> from user-provided evidence.</li>
            <li><b>Safety guardrails:</b> If adverse events or patient-identifiable data are detected, generation is blocked.</li>
          </ul>
          <p class="pd-small-muted">
            Known limitation: keyword-based detection is conservative and not a full compliance solution.
            Always follow local SOPs and PV reporting processes.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_sop_drafting(model: str, temperature: float) -> None:
    st.title("📑 Medical Affairs SOP Drafting Tool")
    render_disclaimer()

    with st.form("sop_form"):
        sop_type = st.selectbox("SOP type", ["MI", "MSL activity", "Insights", "Advisory Boards"])
        geography = st.selectbox("Geography", ["UK", "Ireland", "UK & Ireland", "Other"])
        existing = st.text_area("Existing SOP text (optional)", height=180, placeholder="Paste existing SOP text if available (remove any patient-identifiable content).")
        extra_requirements = st.text_area("Additional requirements (optional)", height=110, placeholder="E.g., required sections, approval workflow, training cadence.")

        run = st.form_submit_button("Draft SOP", use_container_width=True)

    blocked, reasons = detect_ae_or_pii("\n".join([sop_type, geography, existing, extra_requirements]))

    if run:
        if blocked:
            st.markdown(
                "<div class='pd-warn'><b>Generation blocked.</b><br/>" +
                "<br/>".join([st._utils.escape_markdown(r) for r in reasons]) +
                "<br/><br/>Remove AE / patient-identifiable data and try again.</div>",
                unsafe_allow_html=True,
            )
            return

        user_prompt = f"""
SOP type: {sop_type}
Geography: {geography}

Existing SOP text (optional):
{existing}

Additional requirements:
{extra_requirements}
""".strip()

        result = generate_json(
            model=model,
            tool_name="Medical Affairs SOP Drafting Tool",
            system_prompt=TOOL_DEFS["Medical Affairs SOP Drafting Tool"]["system_prompt"],
            user_prompt=user_prompt,
            json_schema_hint=TOOL_DEFS["Medical Affairs SOP Drafting Tool"]["schema"],
            temperature=temperature,
        )
        render_output_block("SOP Draft", result, key_prefix="sop_draft")


# =========================
# Main Router
# =========================

def main() -> None:
    inject_css()

    selected_page, model, temperature = sidebar_nav()

    # Top-level dependency checks (friendly)
    api_key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if not api_key_present:
        st.warning("OPENAI_API_KEY not set. Generation tools will not work until an API key is provided via environment variable.")

    if OpenAI is None:
        st.warning("openai package not available. Install dependencies from requirements.txt to enable generation tools.")

    # Route
    if selected_page == "Home":
        page_home()
    elif selected_page == "Scientific Narrative Generator":
        page_scientific_narrative(model, temperature)
    elif selected_page == "MSL Briefing Pack Generator":
        page_msl_briefing_pack(model, temperature)
    elif selected_page == "Medical Information Response Generator":
        page_mi_response(model, temperature)
    elif selected_page == "Congress & Advisory Board Planner":
        page_congress_adboard(model, temperature)
    elif selected_page == "Insight Capture & Thematic Analysis":
        page_insight_analysis(model, temperature)
    elif selected_page == "Medical Affairs Executive Summary Generator":
        page_exec_summary(model, temperature)
    elif selected_page == "Compliance & Governance Summary":
        page_compliance()
    elif selected_page == "Medical Affairs SOP Drafting Tool":
        page_sop_drafting(model, temperature)
    else:
        st.error("Unknown page selection.")


if __name__ == "__main__":
    main()
