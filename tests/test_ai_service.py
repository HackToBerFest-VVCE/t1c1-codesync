"""
Tests for AI Service integration, model response normalization, and refinement.
"""

from unittest.mock import MagicMock, patch
import pytest
from models import CircuitContext, DiagnosticResult, NextTest
from ai_service import (
    analyze_circuit,
    refine_diagnosis,
    is_image_supported_by_model,
    get_configured_gemma_model,
    GemmaServiceError,
)


def test_is_image_supported_by_model():
    assert is_image_supported_by_model("gemma-4-vision") is True
    assert is_image_supported_by_model("gemma-vl") is True
    assert is_image_supported_by_model("gemma-4") is False
    assert is_image_supported_by_model("gemma-2-9b-it") is False


def test_analyze_circuit_successful_flow(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key_for_testing")
    monkeypatch.setenv("GEMMA_MODEL", "gemma-4")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
    {
      "diagnostic_summary": "Inverted polarity on LED cathode and anode.",
      "observed_facts": ["5V supply rail active", "LED does not illuminate"],
      "assumptions": ["Standard 220 ohm resistor in series"],
      "primary_fault": "Reverse-Biased LED",
      "suspected_component": "LED1",
      "confidence": 88,
      "reasoning": "When an LED is connected in reverse polarity, its PN junction is reverse biased and prevents current flow.",
      "possible_causes": [
        {"cause": "Reversed LED orientation", "likelihood": "High", "reason": "Consistent with 0 mA current draw"}
      ],
      "next_test": {
        "test": "Diode forward voltage check",
        "how_to_test": "Place multimeter in diode mode across LED leads.",
        "expected_result": "1.8V to 2.1V forward drop with red probe on long lead.",
        "if_expected": "LED is functional; re-check wiring polarity.",
        "if_not_expected": "LED is blown open."
      },
      "troubleshooting_steps": [
        {"step": 1, "action": "Invert LED pins on breadboard", "purpose": "Test correct polarity"}
      ],
      "recommended_fix": ["Flip LED orientation 180 degrees"],
      "prevention_tips": ["Observe flat edge denoting cathode lead"],
      "safety_warning": "Disconnect 5V supply before adjusting components.",
      "needs_more_information": false,
      "information_needed": []
    }
    """
    mock_client.models.generate_content.return_value = mock_response

    with patch("ai_service._init_genai_client", return_value=mock_client):
        ctx = CircuitContext(symptoms_text="The LED does not glow at 5V.")
        diag = analyze_circuit(ctx)

        assert diag.primary_fault == "Reverse-Biased LED"
        assert diag.suspected_component == "LED1"
        assert diag.confidence == 88
        assert diag.next_test.test == "Diode forward voltage check"
        assert diag.model_identifier == "gemma-4"


def test_analyze_circuit_malformed_response_raises_friendly_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key_for_testing")
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is not json at all. I am an AI model."
    mock_client.models.generate_content.return_value = mock_response

    with patch("ai_service._init_genai_client", return_value=mock_client):
        ctx = CircuitContext(symptoms_text="Circuit is malfunctioning completely.")
        with pytest.raises(GemmaServiceError) as exc_info:
            analyze_circuit(ctx)
        assert "unexpected response format" in exc_info.value.user_message


def test_refine_diagnosis_successful_flow(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key_for_testing")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
    {
      "previous_fault": "Defective LED",
      "updated_fault": "Reverse-Biased LED Orientation",
      "previous_suspected_component": "LED1",
      "updated_suspected_component": "LED1",
      "previous_confidence": 60,
      "updated_confidence": 94,
      "what_changed": "User measured 2.0V forward drop when probe polarity was swapped, confirming the LED is healthy but installed backwards.",
      "updated_reasoning": "The diode test definitively eliminates the burnt LED hypothesis and confirms reversed polarity in circuit assembly.",
      "updated_recommended_repair": ["Reverse LED pins on breadboard so anode connects to positive supply rail."],
      "next_recommended_action": "Power circuit and verify steady light emission."
    }
    """
    mock_client.models.generate_content.return_value = mock_response

    ctx = CircuitContext(symptoms_text="LED fails to light.")
    prev_diag = DiagnosticResult(
        diagnostic_summary="Preliminary analysis",
        observed_facts=[],
        assumptions=[],
        primary_fault="Defective LED",
        suspected_component="LED1",
        confidence=60,
        reasoning="No light emitted.",
        possible_causes=[],
        next_test=NextTest(
            test="Diode test",
            how_to_test="Measure across LED leads",
            expected_result="2V drop",
            if_expected="Healthy",
            if_not_expected="Burnt",
        ),
        troubleshooting_steps=[],
        recommended_fix=["Replace LED"],
        prevention_tips=[],
        safety_warning="",
        needs_more_information=False,
        information_needed=[],
        model_identifier="gemma-4",
    )

    with patch("ai_service._init_genai_client", return_value=mock_client):
        refined = refine_diagnosis(ctx, prev_diag, "Probe swapped: measured 2.0V in diode mode!")
        assert refined.updated_fault == "Reverse-Biased LED Orientation"
        assert refined.previous_confidence == 60
        assert refined.updated_confidence == 94
        assert "probe polarity was swapped" in refined.what_changed
