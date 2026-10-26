"""
Tests for input validation, prompt synthesis, and error handling.
"""

import os
import pytest
from utils import validate_circuit_input, validate_image_file
from models import CircuitContext, DiagnosticResult, NextTest
from prompts import build_gemma_diagnostic_prompt, build_gemma_refinement_prompt
from ai_service import analyze_circuit, GemmaServiceError


def test_validate_circuit_input_empty():
    valid, msg = validate_circuit_input("")
    assert not valid
    assert "describe what is wrong" in msg.lower()


def test_validate_circuit_input_too_short():
    valid, msg = validate_circuit_input("broken")
    assert not valid
    assert "descriptive" in msg.lower()


def test_validate_circuit_input_valid():
    valid, msg = validate_circuit_input("The 5V regulator output measures only 1.2V and heats up.")
    assert valid
    assert msg == ""


def test_validate_image_file_none():
    valid, msg, img = validate_image_file(None)
    assert valid
    assert msg == ""
    assert img is None


def test_prompt_synthesis_includes_all_measurements():
    ctx = CircuitContext(
        project_name="Audio Preamp",
        supply_voltage="12V DC",
        expected_behaviour="Amplify line-level audio cleanly",
        known_components="NE5532, 10k resistors, 100uF caps",
        symptoms_text="Severe distortion on output channel",
        quick_symptoms=["Intermittent Output", "Wrong Output Voltage"],
        measured_voltage="DC bias = 0.4V on pin 8",
        measured_current="45mA idle",
        measured_resistance="Feedback R = 10.1k",
        other_measurement="Scope shows clipping at 1Vpp",
    )
    prompt = build_gemma_diagnostic_prompt(ctx)
    assert "Audio Preamp" in prompt
    assert "Severe distortion on output channel" in prompt
    assert "DC bias = 0.4V on pin 8" in prompt
    assert "45mA idle" in prompt
    assert "Feedback R = 10.1k" in prompt
    assert "Scope shows clipping at 1Vpp" in prompt
    assert "Intermittent Output" in prompt


def test_refinement_prompt_includes_evidence_loop():
    ctx = CircuitContext(
        symptoms_text="LED fails to light",
        project_name="LED Blinker",
    )
    prev = DiagnosticResult(
        diagnostic_summary="Suspected blown resistor",
        observed_facts=["LED is dark"],
        assumptions=["5V source present"],
        primary_fault="Open Current-Limiting Resistor",
        suspected_component="R1",
        confidence=65,
        reasoning="If R1 is open, no current flows to the LED.",
        possible_causes=[],
        next_test=NextTest(
            test="Voltage Drop Across R1",
            how_to_test="Measure DC voltage between the two resistor leads.",
            expected_result="Approximately 3.0V drop.",
            if_expected="Resistor is conducting normal current.",
            if_not_expected="Zero volts indicates no current path.",
        ),
        troubleshooting_steps=[],
        recommended_fix=["Replace R1 with 220 ohm 0.25W resistor"],
        prevention_tips=[],
        safety_warning="",
        needs_more_information=False,
        information_needed=[],
        model_identifier="gemma-4",
    )
    refine_prompt = build_gemma_refinement_prompt(
        ctx=ctx,
        prev=prev,
        user_test_result="Measured 0.0V across R1, but 5.0V across LED anode and cathode",
    )
    assert "Open Current-Limiting Resistor" in refine_prompt
    assert "Measured 0.0V across R1" in refine_prompt
    assert "Voltage Drop Across R1" in refine_prompt
    assert "STAGE 4: GEMMA 4 REFINEMENT" in refine_prompt


def test_ai_service_missing_api_key_raises_friendly_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMMA_API_KEY", "")
    
    ctx = CircuitContext(symptoms_text="The microcontroller does not boot.")
    with pytest.raises(GemmaServiceError) as exc_info:
        analyze_circuit(ctx)
    assert "Gemma API key is not configured" in exc_info.value.user_message
