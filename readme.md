# PowerDash Medical (Internal MVP)

PowerDash Medical is a **private internal Medical Affairs AI workbench** inspired by the PowerDash HR suite.
It is a **multi-tool Streamlit app** (not a single app) with a shared layout and shared generation engine.

## What this is (and is not)
- ✅ Internal MVP foundation to iterate on
- ✅ Stateless (no database)
- ✅ No file storage of user content
- ✅ No authentication
- ✅ No analytics
- ✅ Streamlit only (Python-first)
- ✅ Conservative Medical Affairs tone (UK & Ireland default)
- ✅ Drafting support only — medical review required

- ❌ Not a public product
- ❌ Not a compliance system
- ❌ Not a pharmacovigilance intake tool

## Tools included
### Core Medical Affairs Tools
1. **Scientific Narrative Generator**
2. **MSL Briefing Pack Generator**
3. **Medical Information Response Generator** (references ONLY from user input; blocks AE/PII)
4. **Congress & Advisory Board Planner**

### Additional Tools
5. **Insight Capture & Thematic Analysis** (session-only insight list)
6. **Medical Affairs Executive Summary Generator**
7. **Compliance & Governance Summary** (static page)
8. **Medical Affairs SOP Drafting Tool**

## Safety & Guardrails
This MVP includes conservative, keyword/regex-based detection:
- **Adverse event / PV indicators** → blocks generation
- **Patient-identifiable data indicators** (email/phone/NHS # + keywords) → blocks generation

If triggered, the app displays a warning and does not call the LLM.

> Known limitation: this is not a full compliance solution. Always follow local SOPs and PV reporting processes.

## Stateless design
- No database
- No saving files to disk (user content)
- No retention of inputs/outputs beyond the Streamlit session
- The Insight Capture tool stores insights only in `st.session_state` (cleared on refresh / new session)

## Setup
### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
