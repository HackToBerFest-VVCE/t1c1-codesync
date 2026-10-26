"""
Tests for JSON extraction, confidence normalization, and safety detection.
"""

import pytest
from utils import clean_and_parse_json, normalize_confidence, detect_safety_concerns
from models import DiagnosticResult, NextTest, PossibleCause, RefinedDiagnosticResult


def test_clean_and_parse_json_pure():
    raw = '{"diagnostic_summary": "Test summary", "confidence": 85}'
    parsed = clean_and_parse_json(raw)
    assert parsed["diagnostic_summary"] == "Test summary"
    assert parsed["confidence"] == 85


def test_clean_and_parse_json_markdown():
    raw = """Here is the diagnostic result:
```json
{
  "diagnostic_summary": "Open circuit fault",
  "confidence": 92
}
```
Please let me know if you need more help."""
    parsed = clean_and_parse_json(raw)
    assert parsed["diagnostic_summary"] == "Open circuit fault"
    assert parsed["confidence"] == 92


def test_clean_and_parse_json_trailing_commas():
    raw = '{"primary_fault": "Burned resistor", "confidence": 70,}'
    parsed = clean_and_parse_json(raw)
    assert parsed["primary_fault"] == "Burned resistor"
    assert parsed["confidence"] == 70


def test_clean_and_parse_json_empty_raises():
    with pytest.raises(ValueError):
        clean_and_parse_json("")


def test_normalize_confidence():
    assert normalize_confidence(95) == 95
    assert normalize_confidence(0.85) == 85
    assert normalize_confidence("78%") == 78
    assert normalize_confidence("0.65") == 65
    assert normalize_confidence(150) == 100
    assert normalize_confidence(-10) == 0
    assert normalize_confidence(None) == 70


def test_detect_safety_concerns_mains():
    warnings = detect_safety_concerns("The circuit is plugged into 230V mains and sparked.")
    assert len(warnings) > 0
    assert any("MAINS" in w for w in warnings)


def test_detect_safety_concerns_thermal():
    warnings = detect_safety_concerns("The power transistor is smoking and very hot.")
    assert len(warnings) > 0
    assert any("THERMAL" in w for w in warnings)


def test_detect_safety_concerns_clean_low_voltage():
    warnings = detect_safety_concerns("5V DC circuit with LED not turning on.")
    assert len(warnings) == 0


def test_diagnostic_result_pydantic_model():
    res = DiagnosticResult(
        diagnostic_summary="LED polarity reversed",
        observed_facts=["5V supply active", "LED dark"],
        assumptions=["Standard red LED with 2.0V forward drop"],
        primary_fault="Reverse-biased LED",
        suspected_component="LED1",
        confidence=88,
        reasoning="An LED is a polar diode; reverse bias blocks current completely.",
        possible_causes=[
            PossibleCause(cause="Inverted polarity", likelihood="High", reason="Common assembly error"),
            PossibleCause(cause="Burnt open LED", likelihood="Low", reason="No sign of overcurrent"),
        ],
        next_test=NextTest(
            test="Diode Mode Test",
            how_to_test="Place red DMM lead on anode, black on cathode in diode mode.",
            expected_result="1.8V to 2.2V forward drop and faint glow.",
            if_expected="LED is operational; check circuit track connections.",
            if_not_expected="LED is defective or backward.",
        ),
        troubleshooting_steps=[],
        recommended_fix=["Reverse LED orientation so long lead connects to positive rail."],
        prevention_tips=["Check flat edge on cathode before soldering."],
        safety_warning="",
        needs_more_information=False,
        information_needed=[],
        model_identifier="gemma-4",
    )
    assert res.confidence == 88
    assert res.primary_fault == "Reverse-biased LED"
    assert res.next_test.test == "Diode Mode Test"
