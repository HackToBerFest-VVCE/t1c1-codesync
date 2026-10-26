"""
CodeSync — Data models for Gemma 4 Circuit Doctor.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class NextTest(BaseModel):
    test: str = Field(..., description="Name or title of the next electrical test")
    how_to_test: str = Field(..., description="Detailed instructions on how and where to measure")
    expected_result: str = Field(..., description="Expected measurement or behavior for a healthy circuit")
    if_expected: str = Field(..., description="Diagnostic conclusion if expected result is observed")
    if_not_expected: str = Field(..., description="Diagnostic conclusion if unexpected result is observed")


class PossibleCause(BaseModel):
    cause: str = Field(..., description="Specific fault hypothesis")
    likelihood: Literal["High", "Medium", "Low"] = Field(..., description="Confidence ranking")
    reason: str = Field(..., description="Evidence-backed reasoning for this cause")


class TroubleshootingStep(BaseModel):
    step: int = Field(..., description="Step number in sequential troubleshooting order")
    action: str = Field(..., description="Action to take")
    purpose: str = Field(..., description="Diagnostic purpose of this step")


class DiagnosticResult(BaseModel):
    diagnostic_summary: str = Field(..., description="High-level summary of the diagnostic finding")
    observed_facts: List[str] = Field(default_factory=list, description="Verified facts from user input")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made about unstated circuit parameters")
    primary_fault: str = Field(..., description="Primary identified circuit fault")
    suspected_component: str = Field(..., description="Suspected faulty component or circuit stage")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score between 0 and 100")
    reasoning: str = Field(..., description="Clear evidence-based diagnostic explanation")
    possible_causes: List[PossibleCause] = Field(default_factory=list, description="Ranked alternative causes")
    next_test: NextTest = Field(..., description="Most informative next electrical test")
    troubleshooting_steps: List[TroubleshootingStep] = Field(default_factory=list, description="Ordered steps")
    recommended_fix: List[str] = Field(default_factory=list, description="Repair or replacement actions")
    prevention_tips: List[str] = Field(default_factory=list, description="Techniques to prevent future recurrence")
    safety_warning: str = Field(default="", description="Safety warnings regarding voltage, heat, or ESD")
    needs_more_information: bool = Field(default=False, description="Flag indicating if evidence is insufficient")
    information_needed: List[str] = Field(default_factory=list, description="Specific parameters needed")
    model_identifier: str = Field(default="gemma-4", description="Actual model identifier used for analysis")


class RefinedDiagnosticResult(BaseModel):
    previous_fault: str = Field(..., description="Fault prior to test verification")
    updated_fault: str = Field(..., description="Re-evaluated fault based on new test evidence")
    previous_suspected_component: str = Field(..., description="Component prior to test")
    updated_suspected_component: str = Field(..., description="Updated suspected component")
    previous_confidence: int = Field(..., ge=0, le=100, description="Confidence before test result")
    updated_confidence: int = Field(..., ge=0, le=100, description="Confidence after test result")
    what_changed: str = Field(..., description="Concise explanation of how the new evidence changed the diagnosis")
    updated_reasoning: str = Field(..., description="Detailed updated technical reasoning")
    updated_recommended_repair: List[str] = Field(default_factory=list, description="Updated repair instructions")
    next_recommended_action: Optional[str] = Field(default=None, description="Next suggested step")
    model_identifier: str = Field(default="gemma-4", description="Actual model identifier used for refinement")


class CircuitContext(BaseModel):
    project_name: Optional[str] = ""
    supply_voltage: Optional[str] = ""
    expected_behaviour: Optional[str] = ""
    known_components: Optional[str] = ""
    additional_notes: Optional[str] = ""
    symptoms_text: str = ""
    quick_symptoms: List[str] = Field(default_factory=list)
    measured_voltage: Optional[str] = ""
    measured_current: Optional[str] = ""
    measured_resistance: Optional[str] = ""
    other_measurement: Optional[str] = ""
    has_image: bool = False
    image_source: Optional[Literal["upload", "camera"]] = None
    image_filename: Optional[str] = None
    image_bytes: Optional[bytes] = None
    image_mime_type: Optional[str] = None
