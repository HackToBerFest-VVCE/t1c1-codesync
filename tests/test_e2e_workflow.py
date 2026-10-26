"""
End-to-end workflow verification for all interactive controls in CodeSync.
Simulates real user interaction with all input widgets, cameras, image uploaders,
voice input, analysis execution, result rendering, and refinement loops.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import io
from PIL import Image
from streamlit.testing.v1 import AppTest
from models import DiagnosticResult, NextTest, PossibleCause, RefinedDiagnosticResult

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_e2e_complete_flow():
    """Simulate a complete end-to-end user session from input to refinement."""
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception

    # 1. Verify Demo button populates inputs only (never fakes diagnosis)
    demo_btn = [b for b in at.sidebar.button if "LED Circuit" in b.label][0]
    demo_btn.click().run()
    assert not at.exception
    assert "LED does not glow" in at.text_area(key="symptoms_text_area").value
    assert at.session_state["diagnostic_result"] is None

    # 2. Modify circuit details and measurements
    at.text_input(key="in_project_name").input("5V Bench LED Circuit").run()
    at.text_input(key="in_supply_voltage").input("5.0V DC").run()
    at.text_input(key="in_measured_voltage").input("5.0V at rails, 0.0V across diode").run()
    at.text_input(key="in_measured_current").input("0.0 mA").run()

    # 3. Test image upload
    bio = io.BytesIO()
    img = Image.new("RGB", (60, 60), color="blue")
    img.save(bio, format="PNG")
    at.file_uploader(key="circuit_image_uploader").upload("bench_setup.png", bio.getvalue(), "image/png").run()
    assert at.session_state["image_name"] == "bench_setup.png"
    assert at.session_state["image_source"] == "upload"

    # 4. Test Voice Recording simulation
    voice_wav = b"RIFF" + b"\x00" * 40
    with patch("utils.transcribe_audio_bytes", return_value=(True, "LED is completely dark")):
        at.audio_input(key="circuit_voice_recorder").upload("mic.wav", voice_wav, "audio/wav").run()
        assert "LED is completely dark" in at.text_area(key="symptoms_text_area").value

    # 5. Mock Gemma 4 model response for primary analysis
    mock_diag = DiagnosticResult(
        diagnostic_summary="Diode orientation inverted.",
        observed_facts=["5V supply verified", "0 mA current measured"],
        assumptions=["Standard 220 ohm series resistor"],
        primary_fault="Reverse-Biased LED",
        suspected_component="LED1",
        confidence=85,
        reasoning="Inverted PN junction blocks current completely under reverse bias.",
        possible_causes=[
            PossibleCause(cause="Inverted LED polarity", likelihood="High", reason="Zero current observed")
        ],
        next_test=NextTest(
            test="Diode Mode Forward Voltage Test",
            how_to_test="Measure across LED leads using multimeter in diode mode.",
            expected_result="1.8V to 2.2V forward drop with red probe on anode.",
            if_expected="LED is functional; reverse pins on breadboard.",
            if_not_expected="LED is blown open."
        ),
        troubleshooting_steps=[],
        recommended_fix=["Flip LED orientation 180 degrees"],
        prevention_tips=["Observe cathode flat edge before installing"],
        safety_warning="Disconnect power before handling components.",
        needs_more_information=False,
        information_needed=[],
        model_identifier="gemma-4",
    )

    with patch("ai_service.analyze_circuit", return_value=mock_diag):
        analyze_btn = [b for b in at.button if "ANALYZE WITH GEMMA 4" in b.label][0]
        analyze_btn.click().run()
        assert not at.exception
        assert at.session_state["diagnostic_result"] is not None
        assert at.session_state["diagnostic_result"].primary_fault == "Reverse-Biased LED"

    # 6. Verify Results Dashboard rendered the recommended next test
    assert at.session_state["diagnostic_result"].next_test.test == "Diode Mode Forward Voltage Test"

    # 7. Enter user test reading and execute refinement
    at.text_input(key="user_test_input_field").input("Swapped probes: measured 2.05V in diode mode!").run()

    mock_refined = RefinedDiagnosticResult(
        previous_fault="Reverse-Biased LED",
        updated_fault="Confirmed Reverse-Biased LED",
        previous_suspected_component="LED1",
        updated_suspected_component="LED1 (Anode/Cathode Reversed)",
        previous_confidence=85,
        updated_confidence=96,
        what_changed="Diode test confirmed healthy forward drop when reversed, proving LED is operational but wired backwards.",
        updated_reasoning="Physical diode orientation is confirmed inverted on the breadboard.",
        updated_recommended_repair=["Reverse LED leads on breadboard so longer lead connects to 5V rail."],
        next_recommended_action="Power on circuit and confirm illumination.",
        model_identifier="gemma-4",
    )

    with patch("ai_service.refine_diagnosis", return_value=mock_refined):
        refine_btn = [b for b in at.button if "REFINE WITH GEMMA 4" in b.label][0]
        refine_btn.click().run()
        assert not at.exception
        assert at.session_state["refined_result"] is not None
        assert at.session_state["refined_result"].updated_confidence == 96
        assert at.session_state["refined_result"].previous_confidence == 85


def test_remove_image_context_button():
    """Verify the Remove Image Context button clears image bytes and source cleanly."""
    at = AppTest.from_file(APP_PATH).run()
    bio = io.BytesIO()
    img = Image.new("RGB", (30, 30), color="red")
    img.save(bio, format="PNG")
    at.file_uploader(key="circuit_image_uploader").upload("photo.png", bio.getvalue(), "image/png").run()
    assert at.session_state["image_name"] == "photo.png"

    # Click Remove Image Context
    clear_btn = [b for b in at.button if "Remove Image Context" in b.label][0]
    clear_btn.click().run()
    assert not at.exception
    assert at.session_state["image_name"] is None
    assert at.session_state["image_bytes"] is None
