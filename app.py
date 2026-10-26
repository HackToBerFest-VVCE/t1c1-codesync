"""
CodeSync — Gemma 4 Circuit Doctor
AI-Powered Electronic Circuit Fault Detection & Troubleshooting

Lead Full-Stack, AI Integration, Electronics Domain, and QA Engineer implementation.
"""

import os
import io
import time
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from models import CircuitContext, DiagnosticResult, RefinedDiagnosticResult
from ai_service import (
    analyze_circuit,
    refine_diagnosis,
    simplify_for_beginner,
    get_configured_gemma_model,
    get_gemma_api_key,
    is_image_supported_by_model,
    GemmaServiceError,
)
from utils import (
    validate_circuit_input,
    validate_image_file,
    detect_safety_concerns,
    transcribe_audio_bytes,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CodeSync — Gemma 4 Circuit Doctor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Futuristic Electronics Laboratory Styling (Custom CSS)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Base Dark Laboratory Canvas */
.stApp {
    background-color: #070B14;
    color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Subtle Circuit Grid Pattern */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-image: 
        radial-gradient(circle at 15% 20%, rgba(0, 240, 255, 0.03) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 40%),
        linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 32px 32px, 32px 32px;
    pointer-events: none;
    z-index: 0;
}

/* Glassmorphism Card Containers */
.glass-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(0, 240, 255, 0.2);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

.glass-card-hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(13, 27, 42, 0.8) 100%);
    border: 1px solid rgba(0, 240, 255, 0.35);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 0 25px rgba(0, 240, 255, 0.12);
}

.glass-card-test {
    background: linear-gradient(135deg, rgba(16, 28, 48, 0.9) 0%, rgba(20, 18, 45, 0.9) 100%);
    border: 1px solid #00F0FF;
    border-radius: 14px;
    padding: 24px;
    margin-top: 15px;
    margin-bottom: 25px;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.18);
}

.glass-card-refinement {
    background: linear-gradient(135deg, rgba(24, 20, 48, 0.85) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid #A78BFA;
    border-radius: 14px;
    padding: 24px;
    margin-top: 15px;
    margin-bottom: 25px;
    box-shadow: 0 0 20px rgba(167, 139, 250, 0.2);
}

/* Badges */
.badge-gemma {
    background: linear-gradient(90deg, #00F0FF 0%, #7928CA 100%);
    color: #070B14;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: inline-block;
    margin-bottom: 8px;
}

.badge-pipeline {
    background: rgba(0, 240, 255, 0.1);
    border: 1px solid rgba(0, 240, 255, 0.3);
    color: #00F0FF;
    font-size: 0.8rem;
    padding: 4px 10px;
    border-radius: 8px;
    font-weight: 600;
}

.badge-likelihood-high {
    background: rgba(239, 68, 68, 0.2);
    border: 1px solid #EF4444;
    color: #FCA5A5;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 6px;
}

.badge-likelihood-med {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid #F59E0B;
    color: #FCD34D;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 6px;
}

.badge-likelihood-low {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10B981;
    color: #6EE7B7;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 6px;
}

/* Safety Alerts */
.safety-banner {
    background: rgba(245, 158, 11, 0.12);
    border-left: 4px solid #F59E0B;
    color: #FDE68A;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 0.85rem;
    margin-bottom: 16px;
}

.safety-hazard-banner {
    background: rgba(239, 68, 68, 0.18);
    border-left: 4px solid #EF4444;
    color: #FECACA;
    padding: 12px 18px;
    border-radius: 6px;
    font-size: 0.9rem;
    margin-bottom: 16px;
    font-weight: 600;
}

/* Metric Display Numbers */
.metric-header {
    color: #94A3B8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.metric-val {
    color: #00F0FF;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.2;
}

.metric-sub {
    color: #64748B;
    font-size: 0.75rem;
    margin-top: 4px;
}

/* Buttons */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}

/* Streamlit Native Modifications */
div[data-testid="stSidebar"] {
    background-color: #0A0F1D;
    border-right: 1px solid rgba(0, 240, 255, 0.1);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "symptoms_text": "",
        "quick_symptoms": [],
        "project_name": "",
        "supply_voltage": "",
        "expected_behaviour": "",
        "known_components": "",
        "additional_notes": "",
        "measured_voltage": "",
        "measured_current": "",
        "measured_resistance": "",
        "other_measurement": "",
        "image_bytes": None,
        "image_source": None,
        "image_name": None,
        "diagnostic_result": None,
        "refined_result": None,
        "user_test_input": "",
        "beginner_text": None,
        "speech_transcription": "",
        "last_processed_audio": None,
        "voice_transcription_msg": None,
        "voice_transcription_error": None,
        "symptoms_text_area": "",
        "in_project_name": "",
        "in_supply_voltage": "",
        "in_expected_behaviour": "",
        "in_known_components": "",
        "in_measured_voltage": "",
        "in_measured_current": "",
        "in_measured_resistance": "",
        "in_other_measurement": "",
        "in_additional_notes": "",
        "user_test_input_field": "",
        "uploader_version": 0,
        "camera_version": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# -----------------------------------------------------------------------------
# Demo Circuit Presets (Inputs Only — Never Fake Diagnoses!)
# -----------------------------------------------------------------------------
def load_demo(demo_id: int):
    """
    Populates input fields with verified educational circuit scenarios.
    Per hackathon rules: NEVER populates a diagnosis. User must click Analyze.
    """
    st.session_state.diagnostic_result = None
    st.session_state.refined_result = None
    st.session_state.beginner_text = None
    st.session_state.user_test_input = ""
    st.session_state.image_bytes = None
    st.session_state.image_source = None
    st.session_state.image_name = None
    st.session_state.uploader_version += 1
    st.session_state.camera_version += 1

    if demo_id == 1:
        proj = "LED Series Resistor Circuit"
        supply = "5 V DC"
        exp = "LED illuminates steadily when 5V power supply is switched on."
        comps = "Standard 5mm Red LED, 220 Ω 0.25W resistor, 5V breadboard power module."
        symp = "The LED does not glow at all even though the 5 V power supply is on and wired to the breadboard rails."
        quicks = ["LED Not Glowing"]
        v_m = "5.02 V at power rails"
        i_m = "0.00 mA"
        r_m = ""
        o_m = ""
        notes = "Double-checked power supply switch is turned on."
    elif demo_id == 2:
        proj = "Resistive Voltage Divider"
        supply = "10 V DC"
        exp = "Divided output voltage across R2 should measure exactly 5.0 V DC (10V / 2)."
        comps = "R1 = 10 kΩ (1%), R2 = 10 kΩ (1%), bench DC power supply."
        symp = "Output voltage is significantly lower than expected: measuring only ~0.8 V instead of the expected 5.0 V."
        quicks = ["Wrong Output Voltage"]
        v_m = "V_in = 10.0 V, V_out = 0.82 V"
        i_m = "0.92 mA from supply"
        r_m = ""
        o_m = ""
        notes = "Resistors were taken from an unsorted lab component bin."
    elif demo_id == 3:
        proj = "Transistor Load Switch"
        supply = "12 V DC (Load) / 3.3 V (Microcontroller GPIO)"
        exp = "12V DC cooling fan turns on when microcontroller sets GPIO pin HIGH (3.3V)."
        comps = "2N2222 NPN BJT, 1 kΩ base resistor, 12V 150mA fan, 1N4007 flyback diode."
        symp = "The 12V fan does not turn on when base drive is applied. Transistor case feels warm to the touch."
        quicks = ["Component Heating", "Motor Not Running"]
        v_m = "V_GPIO = 3.28 V, V_Collector = 11.9 V"
        i_m = "GPIO current = 2.6 mA"
        r_m = ""
        o_m = "Transistor case temp approx 45°C"
        notes = "Microcontroller logic high verified on oscilloscope."

    # Update both canonical state and widget keys
    st.session_state.project_name = proj
    st.session_state.supply_voltage = supply
    st.session_state.expected_behaviour = exp
    st.session_state.known_components = comps
    st.session_state.symptoms_text = symp
    st.session_state.quick_symptoms = quicks
    st.session_state.measured_voltage = v_m
    st.session_state.measured_current = i_m
    st.session_state.measured_resistance = r_m
    st.session_state.other_measurement = o_m
    st.session_state.additional_notes = notes

    # Synchronize widget keys if already present
    st.session_state["symptoms_text_area"] = symp
    st.session_state["in_project_name"] = proj
    st.session_state["in_supply_voltage"] = supply
    st.session_state["in_expected_behaviour"] = exp
    st.session_state["in_known_components"] = comps
    st.session_state["in_measured_voltage"] = v_m
    st.session_state["in_measured_current"] = i_m
    st.session_state["in_measured_resistance"] = r_m
    st.session_state["in_other_measurement"] = o_m
    st.session_state["in_additional_notes"] = notes
    st.session_state["quick_symptoms_select"] = quicks


def handle_voice_recording():
    """
    Callback triggered immediately when audio is recorded via st.audio_input.
    Runs before widgets are instantiated to safely populate the symptom field.
    """
    audio = st.session_state.get("circuit_voice_recorder")
    if audio is not None:
        audio_bytes = audio.getvalue()
        success, result_text = transcribe_audio_bytes(audio_bytes)
        if success:
            existing = st.session_state.get("symptoms_text_area", "").strip()
            if existing:
                updated = f"{existing} {result_text}"
            else:
                updated = result_text
            st.session_state["symptoms_text"] = updated
            st.session_state["symptoms_text_area"] = updated
            st.session_state["voice_transcription_msg"] = f'✓ Transcribed from microphone: "{result_text}"'
            st.session_state["voice_transcription_error"] = None
        else:
            st.session_state["voice_transcription_error"] = result_text
            st.session_state["voice_transcription_msg"] = None


# -----------------------------------------------------------------------------
# Persistent Safety Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="safety-banner">
        ⚡ <strong>Electrical Safety Notice:</strong> CodeSync is an educational diagnostic assistant. 
        Always disconnect circuit power before probing continuity, replacing components, or modifying wiring. 
        Follow appropriate bench safety and ESD handling procedures.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HERO SECTION
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="glass-card-hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <span class="badge-gemma">POWERED BY GEMMA 4</span>
                <h1 style="color: #FFFFFF; margin: 4px 0 6px 0; font-size: 2.3rem; letter-spacing: -0.02em;">
                    CODESYNC — GEMMA 4 CIRCUIT DOCTOR
                </h1>
                <p style="color: #00F0FF; font-size: 1.1rem; font-weight: 500; margin-bottom: 12px;">
                    "From Circuit Symptoms to Smart Solutions."
                </p>
                <p style="color: #94A3B8; font-size: 0.95rem; max-width: 850px; line-height: 1.5; margin-bottom: 16px;">
                    An intelligent electronics engineering assistant powered by the <strong>Gemma 4 Diagnostic Engine</strong>. 
                    Transforms observed circuit malfunctions and bench measurements into accurate fault hypotheses, 
                    recommends decisive electrical tests, and iteratively refines diagnosis as new evidence is gathered.
                </p>
            </div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center; padding-top: 10px; border-top: 1px solid rgba(0, 240, 255, 0.15);">
            <span class="badge-pipeline">01 INPUT</span>
            <span style="color: #64748B;">→</span>
            <span class="badge-pipeline">02 GEMMA 4 ANALYSIS</span>
            <span style="color: #64748B;">→</span>
            <span class="badge-pipeline">03 VERIFY</span>
            <span style="color: #64748B;">→</span>
            <span class="badge-pipeline">04 GEMMA 4 REFINEMENT</span>
            <span style="color: #64748B;">→</span>
            <span class="badge-pipeline">05 SOLUTION</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR: Demo Presets, Model Config & Transparency
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 Demo Scenarios")
    st.caption("Populate inputs with real-world electronics bench test cases (Diagnosis is NEVER faked):")

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        if st.button("LED Circuit", help="5V LED series resistor circuit"):
            load_demo(1)
            st.rerun()
    with col_d2:
        if st.button("Divider", help="10V resistive voltage divider"):
            load_demo(2)
            st.rerun()
    with col_d3:
        if st.button("Transistor", help="NPN transistor switch"):
            load_demo(3)
            st.rerun()

    st.markdown("---")

    st.markdown("### ⚙️ Gemma 4 Configuration")
    configured_model = get_configured_gemma_model()
    has_api_key = bool(get_gemma_api_key() and not get_gemma_api_key().startswith("your_api"))

    st.markdown(f"**Reasoning Engine:** `Gemma 4`")
    st.markdown(f"**Configured Model ID:** `{configured_model}`")

    if has_api_key:
        st.success("API Key: Configured in local environment")
    else:
        st.warning("API Key: Not configured in .env")
        st.caption("Add `GEMINI_API_KEY` to your local `.env` file to run live diagnostic requests.")

    st.markdown("---")

    st.markdown("### 🔍 Model Transparency")
    with st.expander("Gemma 4 Engine Transparency", expanded=False):
        st.markdown(
            f"""
            - **AI Engine:** Google Gemma 4
            - **Configured Model Identifier:** `{configured_model}`
            - **Role:** Electronic Circuit Diagnostic Reasoning Engine
            - **SDK:** Official Google Gen AI Python SDK
            - **Security:** Zero client-side key exposure. Loaded strictly from server `.env`.
            """
        )

    with st.expander("How Gemma Diagnoses Circuits", expanded=False):
        st.markdown(
            """
            ```mermaid
            graph TD
              A[Circuit Evidence] --> B[Gemma 4 Engine]
              B --> C[Initial Fault & Ranked Causes]
              C --> D[Gemma 4 Recommends Next Test]
              D --> E[User Enters Test Result]
              E --> F[Gemma 4 Re-evaluation]
              F --> G[Refined Diagnosis & Repair Fix]
            ```
            Gemma separates verified observed facts from electrical assumptions, identifies likely failing nodes, and selects the single most informative electrical test to confirm or refute the fault.
            """
        )


# -----------------------------------------------------------------------------
# MAIN INPUT SECTION
# -----------------------------------------------------------------------------
st.markdown("### 📋 1. Circuit Malfunction & Evidence Input")

# Detect safety concerns dynamically
if st.session_state.symptoms_text:
    hazards = detect_safety_concerns(st.session_state.symptoms_text + " " + st.session_state.additional_notes)
    for hz in hazards:
        st.markdown(f'<div class="safety-hazard-banner">⚠️ {hz}</div>', unsafe_allow_html=True)

input_col1, input_col2 = st.columns([1.2, 1], gap="medium")

with input_col1:
    st.markdown("#### Primary Malfunction Description *(Required)*")
    
    # Observed problem text area
    symptoms_input = st.text_area(
        label="Describe what is wrong with your circuit:",
        placeholder="The LED does not glow even though the circuit is connected to a 5 V supply.",
        height=140,
        help="Detail the observed symptoms, LED state, power state, heat, or unexpected output.",
        key="symptoms_text_area",
    )
    st.session_state.symptoms_text = symptoms_input

    # Quick Symptoms
    st.markdown("#### Quick Symptoms")
    quick_options = [
        "No Power",
        "LED Not Glowing",
        "Wrong Output Voltage",
        "Component Heating",
        "Excess Current",
        "Intermittent Output",
        "Short Circuit Suspected",
        "Sensor Not Responding",
        "Motor Not Running",
        "Other",
    ]
    selected_quick = st.multiselect(
        "Select matching symptoms to contextualize analysis:",
        options=quick_options,
        default=[s for s in st.session_state.quick_symptoms if s in quick_options],
        help="Quick symptoms help Gemma classify the failure category accurately.",
        key="quick_symptoms_select",
    )
    st.session_state.quick_symptoms = selected_quick

    # Voice Input Section
    with st.expander("🎙 Describe By Voice (Microphone Audio Input)", expanded=True):
        st.caption(
            "Record your circuit problem using your microphone. Your spoken words are transcribed without paid APIs and automatically populated into the symptom field above."
        )

        st.audio_input(
            "🎙 Record Malfunction Symptoms (Click mic icon to record):",
            key="circuit_voice_recorder",
            on_change=handle_voice_recording,
            help="Click to start recording, speak your circuit symptoms, and click stop. Supported across Safari, Chrome, Edge, and Firefox.",
        )

        if st.session_state.get("voice_transcription_msg"):
            st.success(f"{st.session_state.voice_transcription_msg} — populated in symptom field")
        if st.session_state.get("voice_transcription_error"):
            st.warning(f"⚠️ {st.session_state.voice_transcription_error}")

        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 6px;">
                💡 <em>Tip: Speak clearly (e.g., "The LED does not glow even though five volt supply is on"). 
                If microphone permission is denied, enable microphone access in Safari settings (Safari &gt; Settings &gt; Websites &gt; Microphone) or type your symptoms manually above.</em>
            </div>
            """,
            unsafe_allow_html=True,
        )


with input_col2:
    st.markdown("#### Circuit / Schematic Evidence *(Optional)*")
    image_tab1, image_tab2 = st.tabs(["📁 Upload Image", "📸 Take Photo"])

    uploaded_file = None
    with image_tab1:
        uploaded_file = st.file_uploader(
            "Upload circuit schematic, breadboard photo, or PCB layout:",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG (Max 10MB). Circuit diagnosis works fully without an image.",
            key=f"circuit_image_uploader_{st.session_state.uploader_version}",
        )
        if uploaded_file is not None:
            is_valid, msg, img = validate_image_file(uploaded_file)
            if not is_valid:
                st.error(msg)
            else:
                st.session_state.image_bytes = uploaded_file.getvalue()
                st.session_state.image_source = "upload"
                st.session_state.image_name = uploaded_file.name
                st.image(img, caption="Circuit Image Preview (Uploaded)", width="stretch")

    with image_tab2:
        try:
            camera_file = st.camera_input(
                "Capture photo of breadboard or circuit under test:",
                help="Simple stable camera capture. Image is kept in session memory and never permanently stored.",
                key=f"camera_circuit_input_{st.session_state.camera_version}",
            )
            if camera_file is not None:
                is_valid, msg, img = validate_image_file(camera_file)
                if not is_valid:
                    st.error(msg)
                else:
                    st.session_state.image_bytes = camera_file.getvalue()
                    st.session_state.image_source = "camera"
                    st.session_state.image_name = "camera_capture.jpg"
                    st.image(img, caption="Circuit Photo Preview (Camera Capture)", width="stretch")
        except Exception as e:
            st.warning(
                "Camera access is unavailable. Allow camera permission or upload an image instead."
            )

    if st.session_state.image_bytes:
        source_label = "Camera Capture" if st.session_state.image_source == "camera" else "Uploaded Image"
        st.caption(f"✓ Active visual context: **{source_label}**")
        if st.button("Remove Image Context", key="clear_img_btn"):
            st.session_state.image_bytes = None
            st.session_state.image_source = None
            st.session_state.image_name = None
            st.session_state.uploader_version += 1
            st.session_state.camera_version += 1
            st.rerun()


# -----------------------------------------------------------------------------
# CIRCUIT DETAILS & ELECTRICAL MEASUREMENTS
# -----------------------------------------------------------------------------
detail_exp = st.expander("🔧 Circuit Details & Electrical Measurements (Optional)", expanded=True)
with detail_exp:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Circuit Context**")
        st.session_state.project_name = st.text_input(
            "Circuit / Project Name:",
            placeholder="e.g., 5V Power Supply / Audio Preamp",
            key="in_project_name",
        )
        st.session_state.supply_voltage = st.text_input(
            "Nominal Supply Voltage:",
            placeholder="e.g., 5 V DC, 12 V DC, 3.3 V",
            key="in_supply_voltage",
        )
        st.session_state.expected_behaviour = st.text_input(
            "Expected Circuit Behaviour:",
            placeholder="e.g., LED illuminates when power is applied",
            key="in_expected_behaviour",
        )
        st.session_state.known_components = st.text_area(
            "Known Components:",
            placeholder="e.g., LM7805 regulator, 100uF capacitor, red LED, 220 ohm resistor",
            height=68,
            key="in_known_components",
        )

    with col_d2:
        st.markdown("**Electrical Measurements**")
        st.session_state.measured_voltage = st.text_input(
            "Measured Voltage:",
            placeholder="e.g., 5.0 V at rail, 0.2 V across diode",
            key="in_measured_voltage",
        )
        st.session_state.measured_current = st.text_input(
            "Measured Current:",
            placeholder="e.g., 0.0 mA, 120 mA",
            key="in_measured_current",
        )
        st.session_state.measured_resistance = st.text_input(
            "Resistance / Component Value:",
            placeholder="e.g., 220 Ω, open circuit (>20 MΩ)",
            key="in_measured_resistance",
        )
        st.session_state.other_measurement = st.text_input(
            "Other Measurement / Logic State:",
            placeholder="e.g., GPIO pin is HIGH (3.3V), 1 kHz clock active",
            key="in_other_measurement",
        )
        st.session_state.additional_notes = st.text_input(
            "Additional Notes:",
            placeholder="e.g., Tested on breadboard; wiring verified once",
            key="in_additional_notes",
        )


# -----------------------------------------------------------------------------
# PRIMARY ACTION: ANALYZE WITH GEMMA 4
# -----------------------------------------------------------------------------
st.markdown("---")
analyze_btn = st.button("⚡ ANALYZE WITH GEMMA 4", type="primary", use_container_width=True)

if analyze_btn:
    is_valid, err_msg = validate_circuit_input(st.session_state.symptoms_text)
    if not is_valid:
        st.error(err_msg)
    else:
        # Build Context
        ctx = CircuitContext(
            project_name=st.session_state.project_name,
            supply_voltage=st.session_state.supply_voltage,
            expected_behaviour=st.session_state.expected_behaviour,
            known_components=st.session_state.known_components,
            additional_notes=st.session_state.additional_notes,
            symptoms_text=st.session_state.symptoms_text,
            quick_symptoms=st.session_state.quick_symptoms,
            measured_voltage=st.session_state.measured_voltage,
            measured_current=st.session_state.measured_current,
            measured_resistance=st.session_state.measured_resistance,
            other_measurement=st.session_state.other_measurement,
            has_image=bool(st.session_state.image_bytes is not None),
            image_source=st.session_state.image_source,
            image_filename=st.session_state.image_name,
            image_bytes=st.session_state.image_bytes,
        )

        progress_placeholder = st.empty()
        try:
            with progress_placeholder.container():
                st.info("⚙️ Preparing circuit evidence...")
                time.sleep(0.3)
                st.info("📡 Sending diagnostic request to Gemma 4...")
                time.sleep(0.3)
                st.info("🧠 Gemma 4 is analyzing your circuit evidence...")

            result = analyze_circuit(ctx)
            st.session_state.diagnostic_result = result
            st.session_state.refined_result = None
            st.session_state.beginner_text = None
            progress_placeholder.empty()
            st.success("✓ Gemma 4 Circuit Analysis Complete")
        except GemmaServiceError as ge:
            progress_placeholder.empty()
            st.error(ge.user_message)
            if ge.technical_details:
                with st.expander("Technical Error Details"):
                    st.code(ge.technical_details)
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Gemma returned an unexpected response. Please retry the diagnosis.")
            with st.expander("Diagnostic Trace"):
                st.code(str(e))


# -----------------------------------------------------------------------------
# RESULTS DASHBOARD
# -----------------------------------------------------------------------------
diag: DiagnosticResult = st.session_state.diagnostic_result

if diag is not None:
    st.markdown("---")
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <h2 style="color: #00F0FF; margin: 0; font-size: 1.8rem; letter-spacing: -0.01em;">
                ⚡ GEMMA 4 DIAGNOSTIC RESULT
            </h2>
            <span class="badge-gemma">ENGINE: GEMMA 4</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Cards Grid: Primary Fault, Suspected Component, Confidence
    card_col1, card_col2, card_col3 = st.columns([1.2, 1.2, 1])

    with card_col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-header">Most Likely Fault</div>
                <div class="metric-val" style="font-size: 1.3rem; color: #FFFFFF;">
                    {diag.primary_fault}
                </div>
                <div class="metric-sub">Primary failure mode identified by Gemma 4</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-header">Suspected Component / Node</div>
                <div class="metric-val" style="font-size: 1.3rem; color: #00F0FF;">
                    {diag.suspected_component}
                </div>
                <div class="metric-sub">Target component or electrical node</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_col3:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-header">Gemma 4 Confidence</div>
                <div class="metric-val">{diag.confidence}%</div>
                <div class="metric-sub">Evidence-weighted probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(diag.confidence / 100.0)

    # Diagnostic Summary
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-header" style="color: #00F0FF;">Diagnostic Summary</div>
            <p style="font-size: 1.05rem; line-height: 1.5; margin-bottom: 0;">{diag.diagnostic_summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Evidence Used: Observed Facts vs Assumptions
    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div class="metric-header" style="color: #10B981;">✓ Observed Facts (From Input)</div>
            """,
            unsafe_allow_html=True,
        )
        if diag.observed_facts:
            for fact in diag.observed_facts:
                st.markdown(f"- {fact}")
        else:
            st.markdown("- Circuit problem description as entered by user.")
        st.markdown("</div>", unsafe_allow_html=True)

    with ev_col2:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div class="metric-header" style="color: #F59E0B;">⚡ Circuit Assumptions</div>
            """,
            unsafe_allow_html=True,
        )
        if diag.assumptions:
            for asm in diag.assumptions:
                st.markdown(f"- {asm}")
        else:
            st.markdown("- Standard component ratings and breadboard continuity assumed.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Gemma 4 Diagnostic Explanation
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-header" style="color: #A78BFA;">Gemma 4 Diagnostic Explanation</div>
            <p style="font-size: 0.98rem; line-height: 1.6; color: #E2E8F0; margin-bottom: 0;">
                {diag.reasoning}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Alternative Possible Faults
    if diag.possible_causes:
        st.markdown("#### 🔬 Alternative Possible Faults")
        for pc in diag.possible_causes:
            badge_class = (
                "badge-likelihood-high"
                if pc.likelihood == "High"
                else ("badge-likelihood-med" if pc.likelihood == "Medium" else "badge-likelihood-low")
            )
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 600; font-size: 0.95rem; color: #F1F5F9;">{pc.cause}</span>
                        <span class="{badge_class}">{pc.likelihood.upper()}</span>
                    </div>
                    <p style="color: #94A3B8; font-size: 0.85rem; margin: 0;">{pc.reason}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # HERO FEATURE: GEMMA 4 RECOMMENDS THE NEXT ELECTRICAL TEST
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="glass-card-test">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="badge-gemma">HERO FEATURE</span>
                <span style="color: #00F0FF; font-size: 0.85rem; font-weight: 600;">STAGE 03: VERIFY</span>
            </div>
            <h3 style="color: #00F0FF; margin: 0 0 14px 0; font-size: 1.4rem;">
                🎯 GEMMA 4 RECOMMENDS TESTING THIS NEXT
            </h3>
            <div style="background: rgba(0, 240, 255, 0.06); border-left: 3px solid #00F0FF; padding: 12px 16px; border-radius: 4px; margin-bottom: 14px;">
                <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">WHAT TO TEST</div>
                <div style="font-size: 1.15rem; color: #FFFFFF; font-weight: 700; margin-top: 2px;">
                    {diag.next_test.test}
                </div>
            </div>
            <div style="margin-bottom: 14px;">
                <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; margin-bottom: 4px;">HOW / WHERE TO TEST</div>
                <div style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.5;">{diag.next_test.how_to_test}</div>
            </div>
            <div style="margin-bottom: 14px;">
                <div style="font-size: 0.8rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; margin-bottom: 4px;">EXPECTED RESULT (HEALTHY CIRCUIT)</div>
                <div style="color: #6EE7B7; font-size: 0.95rem; font-weight: 600;">{diag.next_test.expected_result}</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 10px;">
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 6px; padding: 10px;">
                    <div style="color: #10B981; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">IF YOU GET EXPECTED RESULT:</div>
                    <div style="color: #E2E8F0; font-size: 0.85rem; margin-top: 4px;">{diag.next_test.if_expected}</div>
                </div>
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 6px; padding: 10px;">
                    <div style="color: #F87171; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">IF YOU DON'T:</div>
                    <div style="color: #E2E8F0; font-size: 0.85rem; margin-top: 4px;">{diag.next_test.if_not_expected}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # INTERACTIVE DIAGNOSTIC REFINEMENT LOOP
    # -------------------------------------------------------------------------
    st.markdown("#### 🔄 Enter Your Test Result & Refine With Gemma 4")
    st.caption("Perform the recommended electrical test on your bench (or enter simulated readings) to complete the feedback loop:")

    refine_col1, refine_col2 = st.columns([3, 1], gap="medium")
    with refine_col1:
        user_test_input = st.text_input(
            label="Enter Actual Electrical Test Reading / Observation:",
            placeholder="e.g., Voltage across LED = 0.05 V (or: 2.1 V drop observed in diode mode)",
            key="user_test_input_field",
        )
        st.session_state.user_test_input = user_test_input

    with refine_col2:
        st.write("")
        st.write("")
        refine_btn = st.button("✦ REFINE WITH GEMMA 4", type="primary", use_container_width=True)

    if refine_btn:
        if not st.session_state.user_test_input.strip():
            st.warning("Please enter your electrical test measurement before refining with Gemma 4.")
        else:
            ctx = CircuitContext(
                project_name=st.session_state.project_name,
                supply_voltage=st.session_state.supply_voltage,
                expected_behaviour=st.session_state.expected_behaviour,
                known_components=st.session_state.known_components,
                additional_notes=st.session_state.additional_notes,
                symptoms_text=st.session_state.symptoms_text,
                quick_symptoms=st.session_state.quick_symptoms,
                measured_voltage=st.session_state.measured_voltage,
                measured_current=st.session_state.measured_current,
                measured_resistance=st.session_state.measured_resistance,
                other_measurement=st.session_state.other_measurement,
                has_image=bool(st.session_state.image_bytes is not None),
                image_source=st.session_state.image_source,
                image_filename=st.session_state.image_name,
            )
            with st.spinner("Gemma 4 is re-evaluating diagnosis using new test evidence..."):
                try:
                    refined = refine_diagnosis(ctx, diag, st.session_state.user_test_input)
                    st.session_state.refined_result = refined
                    st.success("✓ Gemma 4 Diagnostic Refinement Complete")
                except GemmaServiceError as ge:
                    st.error(ge.user_message)
                except Exception as e:
                    st.error("Gemma returned an unexpected refinement response.")
                    with st.expander("Details"):
                        st.code(str(e))

    # Display Refined Diagnosis if Available
    ref: RefinedDiagnosticResult = st.session_state.refined_result
    if ref is not None:
        st.markdown(
            f"""
            <div class="glass-card-refinement">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="badge-gemma" style="background: linear-gradient(90deg, #A78BFA 0%, #00F0FF 100%);">
                        STAGE 04: GEMMA 4 REFINEMENT
                    </span>
                    <span style="color: #A78BFA; font-size: 0.85rem; font-weight: 600;">NEW EVIDENCE INCORPORATED</span>
                </div>
                <h3 style="color: #FFFFFF; margin: 0 0 16px 0; font-size: 1.5rem;">
                    ✨ GEMMA 4 UPDATED DIAGNOSIS
                </h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(167, 139, 250, 0.3); border-radius: 8px; padding: 12px;">
                        <div class="metric-header">Diagnostic Shift</div>
                        <div style="font-size: 1.1rem; color: #94A3B8; text-decoration: line-through;">{ref.previous_fault}</div>
                        <div style="font-size: 1.3rem; color: #00F0FF; font-weight: 700; margin-top: 4px;">
                            ➔ {ref.updated_fault}
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(167, 139, 250, 0.3); border-radius: 8px; padding: 12px;">
                        <div class="metric-header">Confidence Evolution</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #A78BFA;">
                            {ref.previous_confidence}% ➔ {ref.updated_confidence}%
                        </div>
                        <div class="metric-sub">Confidence updated after verified electrical test</div>
                    </div>
                </div>
                <div style="margin-bottom: 14px;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #A78BFA; font-weight: 700; margin-bottom: 4px;">
                        WHAT CHANGED?
                    </div>
                    <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.5;">{ref.what_changed}</p>
                </div>
                <div style="margin-bottom: 14px;">
                    <div style="font-size: 0.8rem; text-transform: uppercase; color: #A78BFA; font-weight: 700; margin-bottom: 4px;">
                        UPDATED REASONING
                    </div>
                    <p style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.5;">{ref.updated_reasoning}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # SOLUTION & REPAIR RECOMMENDATIONS
    # -------------------------------------------------------------------------
    st.markdown("### 🛠 Solution & Repair Recommendations")
    sol_col1, sol_col2 = st.columns(2)

    with sol_col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="metric-header" style="color: #00F0FF;">Recommended Fix Steps</div>
            """,
            unsafe_allow_html=True,
        )
        fixes = ref.updated_recommended_repair if ref else diag.recommended_fix
        for i, fx in enumerate(fixes, start=1):
            st.markdown(f"**{i}.** {fx}")
        st.markdown("</div>", unsafe_allow_html=True)

    with sol_col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="metric-header" style="color: #10B981;">Prevention & Good Engineering Practice</div>
            """,
            unsafe_allow_html=True,
        )
        if diag.prevention_tips:
            for pt in diag.prevention_tips:
                st.markdown(f"- {pt}")
        else:
            st.markdown("- Verify component pinouts and calculate nominal ratings before powering on.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Sequential Troubleshooting Steps
    if diag.troubleshooting_steps:
        with st.expander("Detailed Sequential Troubleshooting Procedure", expanded=False):
            for step in diag.troubleshooting_steps:
                st.markdown(f"**Step {step.step}: {step.action}**")
                st.caption(f"Purpose: {step.purpose}")

    # -------------------------------------------------------------------------
    # OPTIONAL SECONDARY LAYER: OPENAI SIMPLIFICATION FOR BEGINNERS
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 💡 Educational Simplification")
    st.caption("Convert technical circuit diagnostic findings into intuitive beginner analogies (Optional secondary layer):")

    if st.button("💡 SIMPLIFY FOR BEGINNER (Secondary Layer)", key="btn_simplify"):
        with st.spinner("Generating beginner-friendly explanation..."):
            st.session_state.beginner_text = simplify_for_beginner(diag)

    if st.session_state.beginner_text:
        st.markdown(
            f"""
            <div class="glass-card" style="border-color: rgba(245, 158, 11, 0.4);">
                <div class="metric-header" style="color: #F59E0B;">Beginner-Friendly Explanation</div>
                <div style="font-size: 0.95rem; line-height: 1.6; color: #FEF3C7; white-space: pre-wrap;">
{st.session_state.beginner_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # MODEL TRANSPARENCY PANEL FOR JUDGES
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🏛 Judge Audit & Model Transparency")
    with st.expander("🔍 MODEL & ANALYSIS DETAILS (Click to inspect)", expanded=True):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**AI Engine:** `Gemma 4`")
            st.markdown(f"**Actual Configured Model Identifier:** `{diag.model_identifier}`")
            st.markdown(f"**Engine Role:** Electronic circuit diagnostic reasoning engine")
            st.markdown(f"**Request Architecture:** Direct Google Gen AI SDK API Call")

        with col_m2:
            st.markdown("**Evidence Categories Actually Supplied to Model:**")
            st.markdown(f"- {'✓' if st.session_state.symptoms_text else '✗'} Circuit symptoms")
            st.markdown(f"- {'✓' if st.session_state.known_components else '✗'} Component information")
            st.markdown(f"- {'✓' if st.session_state.supply_voltage else '✗'} Nominal supply voltage")
            st.markdown(f"- {'✓' if st.session_state.measured_voltage else '✗'} Measured voltage")
            st.markdown(f"- {'✓' if st.session_state.measured_current else '✗'} Measured current")
            st.markdown(f"- {'✓' if st.session_state.measured_resistance else '✗'} Resistance values")

            # Image analysis transparency
            img_supported = is_image_supported_by_model(diag.model_identifier)
            if st.session_state.image_bytes and img_supported:
                st.markdown(f"- ✓ Circuit image (Sent directly to vision-capable model)")
            elif st.session_state.image_bytes and not img_supported:
                st.markdown(
                    f"- ✗ Circuit image (Preserved for user UI context; configured model `{diag.model_identifier}` operates on text/symptoms/measurements per model honesty rule)"
                )
            else:
                st.markdown(f"- ✗ Circuit image (No image provided; text & measurements used)")

        if ref is not None:
            st.markdown("---")
            st.markdown("**Refinement Loop Verification:**")
            st.markdown(f"- Feedback evidence: `{st.session_state.user_test_input}`")
            st.markdown(f"- Confidence change: `{ref.previous_confidence}%` ➔ `{ref.updated_confidence}%`")
            st.markdown(f"- Diagnosis change: `{ref.previous_fault}` ➔ `{ref.updated_fault}`")


# -----------------------------------------------------------------------------
# PERSISTENT FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.8rem; padding: 15px 0;">
        <strong>CodeSync — Gemma 4 Circuit Doctor</strong> | Team CodeSync Hackathon Project<br/>
        Gemma 4 is the central diagnostic reasoning engine. Disconnect power before modifying live circuits.
    </div>
    """,
    unsafe_allow_html=True,
)
