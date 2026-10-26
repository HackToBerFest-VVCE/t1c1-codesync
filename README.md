# CodeSync

**AI-Powered Electronic Circuit Fault Detection and Troubleshooting Using Gemma 4**

*Product Name:* **CodeSync — Gemma 4 Circuit Doctor**  
*Tagline:* *"From Circuit Symptoms to Smart Solutions."*  
*Team:* **CodeSync**

---

## Problem Statement

Electronics students, beginners, makers, and bench technicians frequently encounter malfunctioning hardware circuits. While it is easy to observe that a circuit is failing (e.g., an LED is dark, a voltage rail is sagging, or a power transistor is overheating), technicians often struggle to answer:

- Which specific component or circuit node is faulty?
- What electrical failure mechanism caused the breakdown?
- Where should multimeter or oscilloscope probes be placed next?
- What reading should normally be expected at that node?
- What does the measured value mean in the context of circuit laws?
- How does new bench evidence change or confirm the initial hypothesis?
- What concrete physical repairs or replacements are needed?
- How can recurrence be prevented through sound electrical design?

Traditional chatbots provide generic explanations without maintaining a disciplined diagnostic methodology. CodeSync bridges this gap by acting as an intelligent electronics diagnostic assistant.

---

## Our Solution

CodeSync transforms circuit troubleshooting into a structured engineering process driven by Google Gemma 4. By analyzing user-reported symptoms, nominal supply voltages, component ratings, and measured test points, CodeSync isolates the probable root cause, ranks alternative failure modes, generates structured troubleshooting steps, and recommends the single most decisive next electrical test.

---

## Core Innovation: Interactive Gemma 4 Diagnostic Loop

Rather than terminating with a static diagnosis, CodeSync implements a closed-loop interactive diagnostic cycle:

```
Circuit Evidence (Symptoms + Measurements + Visual Context)
       ↓
Gemma 4 Reasoning Engine
       ↓
Probable Fault & Ranked Hypotheses
       ↓
Gemma 4 Recommends the Most Decisive Next Electrical Test
       ↓
User Performs / Simulates the Bench Measurement
       ↓
User Enters the Test Reading (e.g., "0.05 V across LED")
       ↓
Gemma 4 Receives the New Electrical Evidence
       ↓
Gemma 4 Re-evaluates Diagnosis & Updates Confidence
       ↓
Refined Diagnosis & Verified Repair Guidance
```

This feedback loop mirrors the exact workflow of an experienced hardware engineer, separating observed facts from assumptions and updating the diagnosis dynamically.

---

## Gemma 4 Integration

Gemma 4 is the central reasoning engine of CodeSync and is responsible for all diagnostic conclusions.

> **Source Code Location:** The genuine Gemma API integration is implemented in [`ai_service.py`](file:///Users/chandanapurushotham/Documents/codesync/ai_service.py), with prompt engineering defined in [`prompts.py`](file:///Users/chandanapurushotham/Documents/codesync/prompts.py) and data models in [`models.py`](file:///Users/chandanapurushotham/Documents/codesync/models.py).

### How the Integration Operates:
1. **Evidence Collection:** The application gathers user-described symptoms, quick symptom classifications, known component lists, nominal supply voltages, and measured electrical values (voltage, current, resistance).
2. **Deterministic Validation:** `utils.py` validates inputs and detects safety hazards (such as AC mains or thermal overcurrent) prior to making an API call.
3. **Structured Model Request:** A strictly formatted engineering prompt is sent via the official Google Gen AI Python SDK (`google-genai`) to the configured Gemma 4 model.
4. **Structured JSON Output:** Gemma produces a comprehensive JSON response adhering to a rigorous schema:
   - Primary fault and suspected component / node
   - Normalized confidence score ($0-100\%$)
   - Evidence separation: verified observed facts vs. circuit assumptions
   - Evidence-based technical reasoning applying Ohm's and Kirchhoff's laws
   - Ranked alternative causes (High / Medium / Low)
   - Decisive next electrical test with exact probe placement and expected outcomes
   - Sequential troubleshooting procedure and repair steps
   - Prevention tips and electrical safety warnings
5. **Interactive Refinement Request:** When the user enters their actual measurement from the recommended test, a second request sends the prior diagnosis, the recommended test, and the new measurement back to Gemma 4. Gemma re-evaluates the hypothesis, updates the confidence score (e.g., $60\% \rightarrow 92\%$), explains what changed, and provides refined repair instructions.

---

## Exact Model Used

- **Configured Model Identifier:** Read dynamically from the environment variable `GEMMA_MODEL` (defaulting to `gemma-4`).
- **API Configuration Status:** *Gemma 4 model/API configuration requires final verification.* Users can configure their verified Gemma 4 model identifier or dedicated hackathon endpoint in `.env`.
- **Honesty Rule:** CodeSync never fabricates diagnostic outputs. If the model API key is unconfigured or unavailable, CodeSync reports the status clearly without replacing it with fake data.

---

## Why Gemma 4

Electronic circuit troubleshooting requires disciplined causal reasoning:
- Understanding semiconductor physics (forward bias, reverse breakdown, saturation, floating inputs).
- Distinguishing verified symptoms from unstated assumptions.
- Formulating falsifiable test hypotheses.
- Evaluating measurement results against theoretical tolerances.

Gemma 4 provides open weights, high reasoning density, and fast inference, making it an ideal core engine for domain-specific engineering diagnostics.

---

## Where Gemma 4 Is Used

Gemma 4 is prominently highlighted across the entire application:
- **Hero Section:** Diagnostic pipeline and Gemma 4 badge.
- **Primary Diagnostic Action:** `⚡ ANALYZE WITH GEMMA 4`.
- **Processing States:** Live truthful telemetry during evidence preparation and model inference.
- **Results Dashboard:** Fault isolation, suspected component, confidence meter, and diagnostic explanation.
- **Hero Feature Card:** `🎯 GEMMA 4 RECOMMENDS TESTING THIS NEXT`.
- **Refinement Section:** `✦ REFINE WITH GEMMA 4` and `GEMMA 4 UPDATED DIAGNOSIS`.
- **Model Transparency Panel:** Audit view disclosing exact model identifier, role, and evidence categories used.

---

## Architecture

```
User
  ↓
Circuit Image / Camera / Context
  ↓
Symptoms + Electrical Measurements
  ↓
CodeSync Input Validation & Safety Detection (utils.py)
  ↓
Gemma 4 via Google Gen AI SDK (ai_service.py)
  ↓
Fault Analysis & Ranked Causes
  ↓
Recommended Diagnostic Test
  ↓
User Enters Actual Test Result
  ↓
Gemma 4 Re-evaluation (Refinement Loop)
  ↓
Refined Diagnosis & Confidence Evolution
  ↓
Repair Guidance & Prevention Tips
```

---

## Features

- **Gemma 4 Central Diagnostic Engine:** Evidence-based circuit fault isolation.
- **Decisive Next-Test Selection:** Multimeter probe placement and outcome branching.
- **Interactive Refinement Loop:** Real-time feedback loop updating diagnosis from new readings.
- **Circuit Schematic & Photo Input:** Dual-mode image support (file upload & webcam capture) with transparent handling.
- **Voice Dictation:** Browser-native Web Speech recognition into symptom descriptions without paid third-party APIs.
- **Interactive Bench Demos:** Three educational presets (LED Series Resistor, Voltage Divider, Transistor Switch) that populate inputs only without faking diagnoses.
- **Electrical Safety Guardrails:** Automatic detection of AC mains hazards, high voltages, and thermal runaways.
- **Model Transparency Panel:** Audit modal designed for hackathon judges to verify inputs and active model configuration.
- **Optional Secondary Beginner Layer:** Educational simplification layer powered by OpenAI (if configured) to translate technical diagnoses into plain English.

---

## Tech Stack

- **Backend & Logic:** Python 3.13
- **User Interface:** Streamlit 1.66
- **AI / SDK:** Google Gen AI Python SDK (`google-genai` v2.33), optional `openai`
- **Data Validation:** Pydantic v2
- **Image Processing:** Pillow (PIL)
- **Environment Management:** python-dotenv
- **Testing:** Pytest

---

## Project Structure

```
codesync/
├── app.py                  # Main Streamlit dashboard & diagnostic workflow
├── ai_service.py           # Core Gemma 4 API client, inference & refinement routines
├── prompts.py              # Structured prompt engineering & JSON schema guidelines
├── models.py               # Pydantic data schemas for diagnoses and evidence
├── utils.py                # JSON extraction, confidence normalization & validation
├── requirements.txt        # Python dependency manifest
├── pytest.ini              # Pytest configuration
├── .env.example            # Environment variable template with placeholders
├── .gitignore              # Protects private secrets and caches
├── README.md               # Complete project documentation
└── tests/
    ├── test_ai_service.py  # Gemma 4 integration, mock flows & refinement tests
    ├── test_parsing.py     # JSON cleaning, markdown extraction & confidence tests
    ├── test_ui.py          # Headless Streamlit AppTest verification
    └── test_validation.py  # Input validation, image checks & safety detection
```

---

## Local Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- `pip` or virtual environment tool

### 2. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your API keys:

```ini
# Google GenAI / Gemma API Key
GEMINI_API_KEY=your_actual_api_key_here

# Exact Model Identifier for Gemma 4 Reasoning Engine
GEMMA_MODEL=gemma-4

# Optional: Custom API Base URL if using a dedicated hackathon endpoint
# GEMMA_API_BASE=https://your-custom-gemma-endpoint.example.com

# Optional: Secondary Beginner Simplification Layer
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## How to Run

Activate your virtual environment and launch the Streamlit application:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Running Automated Tests
```bash
source .venv/bin/activate
pytest
```

---

## Demo Scenarios

Three pre-configured bench scenarios are provided in the sidebar for immediate demonstration. In accordance with hackathon rules, clicking a demo button **only populates input fields**—the user must click `⚡ ANALYZE WITH GEMMA 4` to execute the genuine diagnostic request.

1. **Demo 1 — LED Series Resistor Circuit:**
   - Supply: 5 V DC
   - Symptoms: LED does not glow at all despite active 5V supply rails.
   - Initial Hypothesis: Reverse-biased LED orientation or open-circuit resistor.
   - Decisive Test: Multimeter diode-mode measurement across LED leads.
2. **Demo 2 — Resistive Voltage Divider:**
   - Supply: 10 V DC
   - Symptoms: Divided output measures ~0.8 V instead of the expected 5.0 V.
   - Initial Hypothesis: Incorrect resistor value installed from unsorted bin.
   - Decisive Test: In-circuit resistance check across R1 and R2.
3. **Demo 3 — Transistor Load Switch:**
   - Supply: 12 V DC / 3.3 V GPIO
   - Symptoms: 12V fan does not turn on; transistor package feels warm.
   - Initial Hypothesis: Base drive saturation failure or open collector circuit.
   - Decisive Test: $V_{BE}$ and $V_{CE}$ measurement during active GPIO high.

---

## API Key Security

> **MANDATORY POLICY:** API keys are loaded strictly from server-side environment variables and must **never** be committed to GitHub or exposed in client bundles.

- `.gitignore` explicitly excludes `.env`, `.env.*`, and `.streamlit/secrets.toml`.
- `.env.example` contains placeholders only (`your_api_key_here`).
- Automated codebase scans verify zero hard-coded credentials in the repository.

---

## Safety Disclaimer

> **CAUTION:** CodeSync is an educational diagnostic assistant. Always disconnect circuit power before modifying components, re-soldering, or probing continuity. 
> Circuits involving AC mains ($>50\text{ V}$) or grid-tied power supplies carry severe shock and fire risks and must never be probed without qualified supervision, isolation transformers, and proper CAT-rated instrumentation.

---

## Limitations

- **Text & Measurement Focus:** Direct circuit image analysis requires a multimodal-capable model identifier. When using text-only models, images are preserved for user context while diagnosis is conducted on symptoms and electrical measurements.
- **Physical Probing:** CodeSync suggests tests and interprets results; physical measurements must be executed by the user with actual bench instrumentation.
- **Non-Destructive Testing:** The tool assumes testing begins with non-destructive multimeter checks.

---

## Future Scope

- Direct USB/Bluetooth digital multimeter (DMM) telemetry streaming into the test result field.
- SPICE simulation netlist export for automated simulation verification.
- Interactive PCB trace heat-map visualization.

---

## Team CodeSync

- **Lead Full-Stack & AI Integration Engineer:** CodeSync Team
- **Electronics Domain & QA:** CodeSync Team
- **Hackathon Project:** AI-Powered Electronic Circuit Fault Detection and Troubleshooting Using Gemma 4
