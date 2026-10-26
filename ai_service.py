"""
CodeSync — AI Integration Service for Gemma 4 Circuit Diagnostic Engine.

Central reasoning engine implementation connecting directly to Google GenAI / Gemma API.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv

# Load environment configuration from .env if present
load_dotenv()

from models import (
    DiagnosticResult,
    RefinedDiagnosticResult,
    CircuitContext,
    NextTest,
    PossibleCause,
    TroubleshootingStep,
)
from prompts import (
    build_gemma_diagnostic_prompt,
    build_gemma_refinement_prompt,
    build_beginner_simplification_prompt,
)
from utils import clean_and_parse_json, normalize_confidence

logger = logging.getLogger("codesync.ai_service")


class GemmaServiceError(Exception):
    """Base exception for Gemma diagnostic engine errors with user-friendly messages."""
    def __init__(self, user_message: str, technical_details: Optional[str] = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.technical_details = technical_details


def get_configured_gemma_model() -> str:
    """
    Returns the configured model identifier from environment variables.
    Defaults to 'gemma-4' if not set, following hackathon specification.
    """
    model_name = os.getenv("GEMMA_MODEL", "").strip()
    if not model_name:
        return "gemma-4"
    return model_name


def get_gemma_api_key() -> Optional[str]:
    """
    Retrieves the Google GenAI / Gemma API key from environment variables.
    Checks GEMINI_API_KEY, GOOGLE_API_KEY, and GEMMA_API_KEY.
    """
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMMA_API_KEY")
    )


def is_image_supported_by_model(model_name: str) -> bool:
    """
    Determines if the configured model identifier explicitly supports direct image inputs.
    Text-only models will process text context only, maintaining image transparency.
    """
    model_lower = model_name.lower()
    vision_keywords = ["vision", "multimodal", "vl", "flash", "pro", "ultra"]
    return any(k in model_lower for k in vision_keywords)


def _init_genai_client(api_key: str):
    """
    Initializes the official Google Gen AI SDK client.
    """
    try:
        from google import genai
        custom_base = os.getenv("GEMMA_API_BASE", "").strip()
        if custom_base:
            http_options = {"base_url": custom_base}
            return genai.Client(api_key=api_key, http_options=http_options)
        return genai.Client(api_key=api_key)
    except Exception as e:
        raise GemmaServiceError(
            "Failed to initialize the Google Gen AI client.",
            technical_details=str(e),
        )


def analyze_circuit(context: CircuitContext) -> DiagnosticResult:
    """
    Primary Gemma 4 diagnostic call.
    Translates user circuit symptoms and measurements into a structured diagnostic report.
    """
    api_key = get_gemma_api_key()
    if not api_key or api_key.startswith("your_api_key"):
        raise GemmaServiceError(
            "Gemma API key is not configured. Add it to your local .env file."
        )

    model_name = get_configured_gemma_model()
    client = _init_genai_client(api_key)
    prompt = build_gemma_diagnostic_prompt(context)

    # Determine whether to include image payload
    contents = [prompt]
    supports_multimodal = is_image_supported_by_model(model_name)
    if context.has_image and context.image_bytes and supports_multimodal:
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(context.image_bytes))
            contents.append(img)
        except Exception as e:
            logger.warning("Could not attach image payload: %s", e)

    # Execute model call
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
        )
    except Exception as e:
        err_str = str(e).lower()
        if "api_key" in err_str or "unauthorized" in err_str or "authentication" in err_str or "403" in err_str or "401" in err_str:
            raise GemmaServiceError(
                "Unable to authenticate with the Gemma service. Check the local API configuration.",
                technical_details=str(e),
            )
        elif "quota" in err_str or "rate limit" in err_str or "429" in err_str:
            raise GemmaServiceError(
                "The model request limit has been reached. Please wait briefly and try again.",
                technical_details=str(e),
            )
        elif "not found" in err_str or "model" in err_str and "unavailable" in err_str or "404" in err_str:
            raise GemmaServiceError(
                f"The configured Gemma model ('{model_name}') is currently unavailable or not found on this endpoint.",
                technical_details=str(e),
            )
        elif "connect" in err_str or "network" in err_str or "timeout" in err_str or "nodename" in err_str:
            raise GemmaServiceError(
                "CodeSync could not reach the model service. Check your internet connection and try again.",
                technical_details=str(e),
            )
        else:
            raise GemmaServiceError(
                f"Diagnostic engine request failed: {str(e)}",
                technical_details=str(e),
            )

    # Extract text from response
    raw_text = ""
    if hasattr(response, "text") and response.text:
        raw_text = response.text
    elif hasattr(response, "candidates") and response.candidates:
        first_cand = response.candidates[0]
        if hasattr(first_cand, "content") and hasattr(first_cand.content, "parts"):
            raw_text = "".join(part.text for part in first_cand.content.parts if hasattr(part, "text"))

    if not raw_text.strip():
        raise GemmaServiceError("Gemma returned an unexpected empty response. Please retry the diagnosis.")

    # Parse structured JSON output
    try:
        data = clean_and_parse_json(raw_text)
    except Exception as e:
        logger.error("JSON parsing failure on model response: %s", e)
        raise GemmaServiceError(
            "Gemma returned an unexpected response format. Please retry the diagnosis.",
            technical_details=str(e),
        )

    # Construct and validate DiagnosticResult model
    try:
        next_test_raw = data.get("next_test", {})
        next_test = NextTest(
            test=next_test_raw.get("test", "Multimeter Node Voltage Measurement"),
            how_to_test=next_test_raw.get("how_to_test", "Measure with DC voltmeter across the suspected component."),
            expected_result=next_test_raw.get("expected_result", "Nominal DC operating voltage."),
            if_expected=next_test_raw.get("if_expected", "Component is receiving proper bias; inspect downstream stage."),
            if_not_expected=next_test_raw.get("if_not_expected", "Confirms fault in the current component or supply path."),
        )

        possible_causes = []
        for pc in data.get("possible_causes", []):
            if isinstance(pc, dict):
                likelihood = pc.get("likelihood", "Medium")
                if likelihood not in ["High", "Medium", "Low"]:
                    likelihood = "Medium"
                possible_causes.append(
                    PossibleCause(
                        cause=pc.get("cause", "Potential component failure"),
                        likelihood=likelihood,
                        reason=pc.get("reason", "Consistent with electrical symptoms"),
                    )
                )

        troubleshooting_steps = []
        for i, step in enumerate(data.get("troubleshooting_steps", []), start=1):
            if isinstance(step, dict):
                troubleshooting_steps.append(
                    TroubleshootingStep(
                        step=step.get("step", i),
                        action=step.get("action", ""),
                        purpose=step.get("purpose", ""),
                    )
                )

        normalized_conf = normalize_confidence(data.get("confidence", 75))

        return DiagnosticResult(
            diagnostic_summary=data.get("diagnostic_summary", "Electronic circuit fault analysis completed."),
            observed_facts=data.get("observed_facts", []),
            assumptions=data.get("assumptions", []),
            primary_fault=data.get("primary_fault", "Undetermined circuit anomaly"),
            suspected_component=data.get("suspected_component", "Circuit block under test"),
            confidence=normalized_conf,
            reasoning=data.get("reasoning", "Evidence points toward an electrical discontinuity or component failure."),
            possible_causes=possible_causes,
            next_test=next_test,
            troubleshooting_steps=troubleshooting_steps,
            recommended_fix=data.get("recommended_fix", ["Inspect physical solder joints and component orientation."]),
            prevention_tips=data.get("prevention_tips", ["Verify voltage and power ratings before powering on."]),
            safety_warning=data.get("safety_warning", "Disconnect power supply before replacing components."),
            needs_more_information=bool(data.get("needs_more_information", False)),
            information_needed=data.get("information_needed", []),
            model_identifier=model_name,
        )
    except Exception as e:
        logger.error("Error constructing DiagnosticResult: %s", e)
        raise GemmaServiceError(
            "Gemma returned an unexpected response structure. Please retry the diagnosis.",
            technical_details=str(e),
        )


def refine_diagnosis(
    context: CircuitContext,
    previous_diagnosis: DiagnosticResult,
    user_test_result: str,
) -> RefinedDiagnosticResult:
    """
    Refinement step: Sends new electrical test results back to Gemma 4
    to update the diagnosis and evaluate confidence changes.
    """
    api_key = get_gemma_api_key()
    if not api_key or api_key.startswith("your_api_key"):
        raise GemmaServiceError("Gemma API key is not configured. Add it to your local .env file.")

    model_name = get_configured_gemma_model()
    client = _init_genai_client(api_key)
    prompt = build_gemma_refinement_prompt(context, previous_diagnosis, user_test_result)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
        )
    except Exception as e:
        err_str = str(e).lower()
        if "api_key" in err_str or "unauthorized" in err_str or "403" in err_str or "401" in err_str:
            raise GemmaServiceError("Unable to authenticate with the Gemma service. Check the local API configuration.")
        elif "quota" in err_str or "429" in err_str:
            raise GemmaServiceError("The model request limit has been reached. Please wait briefly and try again.")
        elif "connect" in err_str or "network" in err_str or "nodename" in err_str:
            raise GemmaServiceError("CodeSync could not reach the model service. Check your internet connection and try again.")
        else:
            raise GemmaServiceError(f"Refinement request failed: {str(e)}", technical_details=str(e))

    raw_text = ""
    if hasattr(response, "text") and response.text:
        raw_text = response.text
    elif hasattr(response, "candidates") and response.candidates:
        first_cand = response.candidates[0]
        if hasattr(first_cand, "content") and hasattr(first_cand.content, "parts"):
            raw_text = "".join(part.text for part in first_cand.content.parts if hasattr(part, "text"))

    if not raw_text.strip():
        raise GemmaServiceError("Gemma returned an unexpected empty response during refinement. Please retry.")

    try:
        data = clean_and_parse_json(raw_text)
    except Exception as e:
        raise GemmaServiceError("Gemma returned an unparseable refinement response. Please retry.", technical_details=str(e))

    try:
        updated_conf = normalize_confidence(data.get("updated_confidence", previous_diagnosis.confidence + 10))
        return RefinedDiagnosticResult(
            previous_fault=previous_diagnosis.primary_fault,
            updated_fault=data.get("updated_fault", previous_diagnosis.primary_fault),
            previous_suspected_component=previous_diagnosis.suspected_component,
            updated_suspected_component=data.get("updated_suspected_component", previous_diagnosis.suspected_component),
            previous_confidence=previous_diagnosis.confidence,
            updated_confidence=updated_conf,
            what_changed=data.get("what_changed", "The new measurement provided concrete evidence to update the fault analysis."),
            updated_reasoning=data.get("updated_reasoning", "Updated reasoning incorporates the verified electrical test result."),
            updated_recommended_repair=data.get("updated_recommended_repair", previous_diagnosis.recommended_fix),
            next_recommended_action=data.get("next_recommended_action"),
            model_identifier=model_name,
        )
    except Exception as e:
        raise GemmaServiceError("Failed to build updated diagnosis from Gemma output.", technical_details=str(e))


def simplify_for_beginner(diagnosis: DiagnosticResult) -> str:
    """
    Optional secondary layer: Calls OpenAI if configured to simplify
    Gemma's technical findings into beginner-friendly terminology.
    Gemma remains the sole diagnostic engine.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key or openai_key.startswith("your_openai"):
        return (
            "Beginner simplification layer (optional OpenAI) is not configured. "
            "Add OPENAI_API_KEY to your .env to enable friendly explanations."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        prompt = build_beginner_simplification_prompt(diagnosis)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content or "No simplified text generated."
    except Exception as e:
        return f"Could not generate beginner simplification: {str(e)}"
