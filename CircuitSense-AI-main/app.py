"""CircuitSense AI — Streamlit Application.

AI-powered electronic circuit troubleshooting and learning assistant for ECE students.
Built with Streamlit and Gemma 4 on Ollama.
"""

import json
from typing import Any, Dict, List, Optional

from PIL import Image
import streamlit as st

from gemma_client import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_DESCRIPTION_LENGTH,
    MAX_IMAGE_SIZE_BYTES,
    MAX_OPTIONAL_FIELD_LENGTH,
    CircuitSenseError,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
    ValidationError,
    diagnose_circuit,
    get_ollama_config,
)

# Circuit type options for user selection
CIRCUIT_TYPE_OPTIONS: List[str] = [
    "Not specified",
    "LED circuit",
    "Power supply",
    "Amplifier",
    "Digital logic",
    "Sensor circuit",
    "Motor driver",
    "Other",
]


def setup_page() -> None:
    """Configure Streamlit page properties and responsive styling."""
    st.set_page_config(
        page_title="CircuitSense AI — Circuit Troubleshooting & Learning",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Scoped custom styling for professional engineering look
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: #1E293B;
        }
        .tagline {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .safety-callout {
            border-left: 4px solid #EF4444;
            background-color: #FEF2F2;
            padding: 0.75rem 1rem;
            border-radius: 4px;
            margin: 0.75rem 0;
            color: #991B1B;
        }
        .concept-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.5rem;
        }
        .instruction-box {
            background-color: #F1F5F9;
            border: 1px dashed #CBD5E1;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            color: #475569;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(base_url: str, model_name: str) -> None:
    """Render sidebar with educational tips and runtime configuration."""
    with st.sidebar:
        st.subheader("⚡ CircuitSense AI")
        st.caption("AI-powered circuit fault detection and learning for ECE students.")

        st.divider()

        st.markdown("### 🔌 Runtime Configuration")
        st.text(f"Model: {model_name}")
        st.text(f"Endpoint: {base_url}")
        st.caption("Configured via environment variables OLLAMA_BASE_URL and OLLAMA_MODEL.")

        st.divider()

        st.markdown("### 📸 Best Practices for Photos")
        st.markdown(
            "- **Lighting:** Use direct, glare-free light.\n"
            "- **Angle:** Take a top-down or slight 45° angle showing breadboard rails.\n"
            "- **Clarity:** Ensure resistor bands and IC pin 1 notches are in focus.\n"
            "- **Wiring:** Avoid wires completely draping over ICs or terminals."
        )

        st.divider()

        st.markdown("### ⚠️ Electrical Safety")
        st.markdown(
            "1. **Always disconnect power** before touching conductors or moving components.\n"
            "2. Never short batteries or power supplies.\n"
            "3. If components overheat or smoke, immediately disconnect power and seek lab instructor assistance."
        )


def validate_structure(data: Any) -> bool:
    """Confirm diagnostic response is a dictionary with minimum usable content."""
    if not isinstance(data, dict):
        return False
    return "summary" in data or "possible_faults" in data


def render_safety_warnings(warnings: Any) -> None:
    """Display safety warnings prominently, highlighting severe hazards."""
    if not isinstance(warnings, list) or not warnings:
        return

    st.markdown("#### ⚠️ Safety Precautions")
    critical_keywords = ("mains", "high voltage", "smoke", "burning", "fire", "overheat", "swollen", "hazard")

    for warn in warnings:
        if isinstance(warn, str) and warn.strip():
            text = warn.strip()
            lower_text = text.lower()
            if any(k in lower_text for k in critical_keywords):
                st.error(f"🚨 **CRITICAL HAZARD:** {text}")
            else:
                st.warning(text)


def render_fault_card(fault_data: Any, index: int) -> None:
    """Render a single candidate fault card with qualitative confidence and evidence."""
    if not isinstance(fault_data, dict):
        return

    fault_name = fault_data.get("fault")
    if not isinstance(fault_name, str) or not fault_name.strip():
        fault_name = f"Hypothesis {index + 1}"
    else:
        fault_name = fault_name.strip()

    confidence = fault_data.get("confidence")
    if not isinstance(confidence, str) or confidence not in ("Low", "Medium", "High"):
        confidence = "Low"

    # Qualitative badge
    if confidence == "High":
        conf_badge = ":green[**Confidence: High**]"
    elif confidence == "Medium":
        conf_badge = ":orange[**Confidence: Medium**]"
    else:
        conf_badge = ":blue[**Confidence: Low**]"

    with st.expander(f"Candidate Fault {index + 1}: {fault_name}", expanded=(index == 0)):
        st.markdown(f"**Fault:** {fault_name}")
        st.markdown(f"**Qualitative Confidence:** {conf_badge}")

        evidence = fault_data.get("evidence")
        if isinstance(evidence, list) and evidence:
            st.markdown("**Visual & Textual Evidence:**")
            for ev in evidence:
                if isinstance(ev, str) and ev.strip():
                    st.markdown(f"- {ev.strip()}")

        verification = fault_data.get("verification_steps")
        if isinstance(verification, list) and verification:
            st.markdown("**Safe Verification Steps:**")
            for step_idx, step in enumerate(verification, start=1):
                if isinstance(step, str) and step.strip():
                    st.markdown(f"{step_idx}. {step.strip()}")


def render_results(diagnosis: Any) -> None:
    """Render the full diagnostic output defensively without assumptions."""
    if not validate_structure(diagnosis):
        st.error("Received an improperly formatted diagnostic result from the model. Please retry the analysis.")
        return

    st.markdown("### 📊 Diagnostic Results")

    # 1. Summary
    summary = diagnosis.get("summary")
    if isinstance(summary, str) and summary.strip():
        st.markdown("#### 📋 Assessment Summary")
        st.info(summary.strip())

    # 2. Safety Warnings
    render_safety_warnings(diagnosis.get("safety_warnings"))

    # 3. Possible Faults
    st.markdown("#### 🔍 Identified Potential Faults")
    possible_faults = diagnosis.get("possible_faults")
    if isinstance(possible_faults, list) and possible_faults:
        valid_faults = [f for f in possible_faults if isinstance(f, dict)]
        if valid_faults:
            for idx, fault in enumerate(valid_faults[:3]):
                render_fault_card(fault, idx)
        else:
            st.write("No specific faults could be confirmed from the provided evidence.")
    else:
        st.write("The provided visual evidence and description do not support a specific fault candidate.")

    # 4. Unknowns
    unknowns = diagnosis.get("unknowns")
    if isinstance(unknowns, list) and unknowns:
        valid_unknowns = [u for u in unknowns if isinstance(u, str) and u.strip()]
        if valid_unknowns:
            st.markdown("#### ❓ Unknowns & Unconfirmed Details")
            st.caption("Information that cannot be established from the image or description:")
            for item in valid_unknowns:
                st.markdown(f"- {item.strip()}")

    # 5. Safe Next Steps
    safe_steps = diagnosis.get("safe_next_steps")
    if isinstance(safe_steps, list) and safe_steps:
        valid_steps = [s for s in safe_steps if isinstance(s, str) and s.strip()]
        if valid_steps:
            st.markdown("#### 🛠️ Recommended Safe Actions")
            for idx, step in enumerate(valid_steps, start=1):
                st.markdown(f"{idx}. {step.strip()}")

    # 6. Learning Explanation
    learning = diagnosis.get("learning_explanation")
    if isinstance(learning, str) and learning.strip():
        st.markdown("#### 💡 Electronics Learning Concept")
        st.markdown(f"> {learning.strip()}")

    # 7. Safety Disclaimer
    st.divider()
    st.caption(
        "⚡ **Educational Safety Notice:** CircuitSense AI diagnoses are educational hypotheses, "
        "not guaranteed repairs. Solderless breadboards contain internal conductive clips and split power rails. "
        "Always isolate power before handling wires or components."
    )

    # 8. Report Download
    report_json = json.dumps(diagnosis, indent=2, ensure_ascii=False)
    st.download_button(
        label="📥 Download Diagnostic Report (JSON)",
        data=report_json,
        file_name="circuitsense_report.json",
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    """Main application loop."""
    setup_page()

    base_url, model_name = get_ollama_config()
    render_sidebar(base_url=base_url, model_name=model_name)

    # Branded Header
    st.markdown('<div class="main-header">⚡ CircuitSense AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tagline">AI-powered electronic circuit troubleshooting and learning for ECE students</div>',
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "diagnosis" not in st.session_state:
        st.session_state["diagnosis"] = None

    # Two-column layout
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📷 Circuit Input")

        # 1. Circuit Image Upload
        uploaded_file = st.file_uploader(
            "Upload Circuit Photo *",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supported formats: PNG, JPEG, WebP. Maximum file size: 10 MB.",
        )

        image_bytes: Optional[bytes] = None

        if uploaded_file is not None:
            # Enforce 10 MB size limit
            if uploaded_file.size > MAX_IMAGE_SIZE_BYTES:
                st.error(f"Image exceeds the 10 MB limit ({uploaded_file.size / (1024*1024):.1f} MB). Please choose a smaller image.")
            else:
                image_bytes = uploaded_file.getvalue()
                st.image(
                    uploaded_file,
                    caption="Circuit Preview",
                    use_container_width=True,
                )
        else:
            st.markdown(
                """
                <div class="instruction-box">
                    <strong>No photo uploaded yet</strong><br>
                    Upload a clear image of your breadboard or PCB circuit to start troubleshooting.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. Problem Description
        problem_description = st.text_area(
            "Observed Problem / Symptoms *",
            placeholder=(
                "Describe what is happening vs. what you expect.\n"
                "Examples:\n"
                "- The red LED does not light up when the 9V battery is connected.\n"
                "- The output voltage drops to 0.8V under load.\n"
                "- Motor vibrates but does not spin when the switch is closed."
            ),
            help=f"Required. Maximum {MAX_DESCRIPTION_LENGTH} characters. Concrete symptoms improve diagnostic precision.",
            max_chars=MAX_DESCRIPTION_LENGTH,
            height=120,
        )

        # 3. Optional Circuit Context
        selected_circuit_type = st.selectbox(
            "Circuit Type / Category (Optional)",
            options=CIRCUIT_TYPE_OPTIONS,
            index=0,
            help="Select the category that best matches your circuit's intended purpose.",
        )
        # Map "Not specified" to empty string
        circuit_type = "" if selected_circuit_type == "Not specified" else selected_circuit_type

        components = st.text_input(
            "Known Components & Values (Optional)",
            placeholder="e.g., 9V battery, 330Ω resistor, 5mm red LED, 1N4007 diode",
            help=f"Optional. Maximum {MAX_OPTIONAL_FIELD_LENGTH} characters.",
            max_chars=MAX_OPTIONAL_FIELD_LENGTH,
        )

        st.caption("* Required fields")

        # 4. Submit Button
        submit_pressed = st.button(
            "⚡ Diagnose Circuit",
            type="primary",
            use_container_width=True,
        )

        if submit_pressed:
            # Client-side validations before network dispatch
            if not image_bytes:
                st.error("Please upload a circuit image (PNG, JPEG, or WebP) before submitting.")
            elif not problem_description or not problem_description.strip():
                st.error("Please provide an observed problem description.")
            else:
                with st.spinner("Analyzing circuit layout and diagnosing faults with Gemma 4..."):
                    try:
                        result = diagnose_circuit(
                            image_bytes=image_bytes,
                            problem_description=problem_description,
                            circuit_type=circuit_type,
                            components=components,
                            timeout=DEFAULT_TIMEOUT_SECONDS,
                        )
                        st.session_state["diagnosis"] = result
                        st.success("Analysis complete!")
                    except ValidationError as exc:
                        st.error(f"⚠️ Validation Error: {exc}")
                    except OllamaConnectionError as exc:
                        st.error(
                            f"🔌 Connection Error: Could not connect to Ollama at {base_url}. "
                            "Please ensure the Ollama service is running."
                        )
                    except OllamaTimeoutError:
                        st.error(
                            "⏳ Request Timeout: The model took longer than 180 seconds to respond. "
                            "Please verify your connection and try again."
                        )
                    except OllamaResponseError as exc:
                        st.error(f"❌ Model Response Error: {exc}. Please verify inputs and retry.")
                    except CircuitSenseError as exc:
                        st.error(f"⚠️ Operational Error: {exc}")
                    except Exception as exc:
                        st.error(f"⚠️ Unexpected Error: {exc}")

    # Right column: Results display
    with col_results:
        if st.session_state["diagnosis"] is not None:
            render_results(st.session_state["diagnosis"])
        else:
            st.markdown("### 📊 Diagnostic Results")
            st.info(
                "Upload a circuit photo and provide your symptom description on the left, "
                "then click **Diagnose Circuit** to run the AI fault analysis."
            )
            st.markdown(
                """
                #### What CircuitSense AI checks:
                - **Physical Connectivity:** Breadboard row alignment, power rail connections, open tracks.
                - **Polarity & Orientation:** Diodes, LEDs, electrolytic capacitors, IC pin 1 orientation.
                - **Current Limiting & Protection:** Resistors in series with LEDs, protection diodes.
                - **Safe Verification Steps:** Multimeter checks and continuity tests to isolate issues safely.
                """
            )


if __name__ == "__main__":
    main()
