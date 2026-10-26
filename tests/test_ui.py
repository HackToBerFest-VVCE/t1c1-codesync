"""
Headless UI and Streamlit workflow tests for CodeSync Gemma 4 Circuit Doctor.
"""

import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from models import DiagnosticResult, NextTest, PossibleCause, RefinedDiagnosticResult

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_initial_load():
    """Verify app initializes without exceptions."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    # Ensure primary analyze button exists
    analyze_btns = [b for b in at.button if "ANALYZE WITH GEMMA 4" in b.label]
    assert len(analyze_btns) == 1


def test_demo_circuits_populate_inputs_only():
    """Verify demo buttons populate inputs only and never fake a diagnosis."""
    at = AppTest.from_file(APP_PATH).run()
    
    # Click Demo 1 (LED Circuit)
    demo_btn = [b for b in at.sidebar.button if "LED Circuit" in b.label][0]
    demo_btn.click().run()
    assert not at.exception
    
    # Check that symptoms text area was populated
    symptom_ta = at.text_area(key="symptoms_text_area")
    assert "LED does not glow" in symptom_ta.value
    
    # Ensure diagnosis result remains None (NEVER fakes a diagnosis)
    assert at.session_state["diagnostic_result"] is None


def test_empty_symptom_validation():
    """Verify clicking Analyze with empty symptoms triggers validation error."""
    at = AppTest.from_file(APP_PATH).run()
    
    # Clear symptom field
    symptom_ta = at.text_area(key="symptoms_text_area")
    symptom_ta.input("").run()
    
    # Click Analyze
    analyze_btn = [b for b in at.button if "ANALYZE WITH GEMMA 4" in b.label][0]
    analyze_btn.click().run()
    
    # Should display validation error
    assert len(at.error) > 0
    assert any("describe what is wrong" in err.value.lower() for err in at.error)


def test_image_upload_workflow():
    """Verify image upload parses correctly, populates session state, and renders without error."""
    import io
    from PIL import Image

    bio = io.BytesIO()
    img = Image.new("RGB", (64, 64), color="green")
    img.save(bio, format="PNG")
    png_bytes = bio.getvalue()

    at = AppTest.from_file(APP_PATH).run()
    uploader = at.file_uploader(key="circuit_image_uploader")
    uploader.upload("circuit_schematic.png", png_bytes, "image/png").run()

    assert not at.exception
    assert at.session_state["image_name"] == "circuit_schematic.png"
    assert at.session_state["image_source"] == "upload"
    assert at.session_state["image_bytes"] == png_bytes


def test_results_dashboard_renders_cleanly():
    """Verify results cards and hero next test feature render when diagnostic_result is populated."""
    at = AppTest.from_file(APP_PATH).run()
    
    sample_result = DiagnosticResult(
        diagnostic_summary="Open circuit detected on pull-up resistor.",
        observed_facts=["Microcontroller input floating", "3.3V supply rail is active"],
        assumptions=["Standard breadboard wiring"],
        primary_fault="Open-Circuit Pull-Up Resistor",
        suspected_component="R_PU (10k)",
        confidence=84,
        reasoning="A floating CMOS input causes random logic toggling without a deterministic pull-up.",
        possible_causes=[
            PossibleCause(cause="Missing pull-up", likelihood="High", reason="Line floats at 1.1V"),
            PossibleCause(cause="Defective breadboard track", likelihood="Low", reason="Contact resistance high"),
        ],
        next_test=NextTest(
            test="DC Node Voltage on MCU Pin 4",
            how_to_test="Connect DMM red probe to MCU pin 4 and black probe to ground.",
            expected_result="3.3 V DC steady state.",
            if_expected="Pull-up is operating; check MCU firmware.",
            if_not_expected="Pull-up path is open or shorted to ground.",
        ),
        troubleshooting_steps=[],
        recommended_fix=["Replace R_PU with verified 10k resistor"],
        prevention_tips=["Use internal pull-up resistor in firmware if available."],
        safety_warning="",
        needs_more_information=False,
        information_needed=[],
        model_identifier="gemma-4",
    )
    
    # Inject result into session state and rerun
    at.session_state["diagnostic_result"] = sample_result
    at.run()
    assert not at.exception
    
    # Check that refine button is rendered
    refine_btns = [b for b in at.button if "REFINE WITH GEMMA 4" in b.label]
    assert len(refine_btns) == 1


def test_refined_result_renders_cleanly():
    """Verify updated diagnosis and confidence evolution render when refined_result is populated."""
    at = AppTest.from_file(APP_PATH).run()
    
    sample_result = DiagnosticResult(
        diagnostic_summary="Suspected faulty LED.",
        observed_facts=["LED is dark"],
        assumptions=[],
        primary_fault="Defective LED",
        suspected_component="LED1",
        confidence=60,
        reasoning="LED does not conduct current.",
        possible_causes=[],
        next_test=NextTest(
            test="Diode drop test",
            how_to_test="Measure across LED leads in diode mode.",
            expected_result="1.8V to 2.2V drop.",
            if_expected="LED healthy.",
            if_not_expected="LED open or reverse-biased.",
        ),
        troubleshooting_steps=[],
        recommended_fix=["Replace LED"],
        prevention_tips=[],
        safety_warning="",
        needs_more_information=False,
        information_needed=[],
        model_identifier="gemma-4",
    )
    
    sample_refined = RefinedDiagnosticResult(
        previous_fault="Defective LED",
        updated_fault="Reverse-Biased LED Orientation",
        previous_suspected_component="LED1",
        updated_suspected_component="LED1 (Anode/Cathode Reversed)",
        previous_confidence=60,
        updated_confidence=92,
        what_changed="Diode test revealed normal 1.9V forward drop when probes were inverted, proving the LED is healthy but installed backwards.",
        updated_reasoning="The component has not burned out; the anode is currently wired to ground instead of the current-limiting resistor.",
        updated_recommended_repair=["Rotate LED by 180 degrees so the longer lead (anode) connects to the resistor."],
        next_recommended_action="Power on 5V supply and verify illumination.",
        model_identifier="gemma-4",
    )
    
    at.session_state["diagnostic_result"] = sample_result
    at.session_state["refined_result"] = sample_refined
    at.run()
    assert not at.exception
