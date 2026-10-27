"""CircuitSense AI — Unit Tests for Prompts, Inputs, and Gemma Client.

These tests verify input validation, image handling, prompt construction,
and mocked API interactions without requiring external network access.
"""

import io
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import pytest
import requests

from gemma_client import (
    MAX_DESCRIPTION_LENGTH,
    MAX_IMAGE_SIZE_BYTES,
    MAX_OPTIONAL_FIELD_LENGTH,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
    ValidationError,
    build_chat_payload,
    diagnose_circuit,
    get_ollama_config,
    parse_model_response,
    validate_image_bytes,
    validate_text_inputs,
)
from prompts import (
    DIAGNOSTIC_SCHEMA,
    DIAGNOSTIC_SYSTEM_PROMPT,
    build_user_prompt,
)


# Helper functions for generating test fixtures
def create_test_image(format_name: str, size: tuple[int, int] = (20, 20)) -> bytes:
    """Generate in-memory image bytes for testing."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color="teal")
    img.save(buf, format=format_name)
    return buf.getvalue()


@pytest.fixture
def valid_png_bytes() -> bytes:
    return create_test_image("PNG")


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    return create_test_image("JPEG")


@pytest.fixture
def valid_webp_bytes() -> bytes:
    return create_test_image("WEBP")


@pytest.fixture
def sample_diagnostic_dict() -> dict:
    return {
        "summary": "LED circuit is non-functional due to open circuit.",
        "possible_faults": [
            {
                "fault": "Reverse LED polarity",
                "evidence": ["LED longer lead (anode) is connected to ground rail"],
                "confidence": "High",
                "verification_steps": ["Disconnect power and swap LED orientation"],
            }
        ],
        "unknowns": ["Exact resistor value from color bands"],
        "safe_next_steps": ["Disconnect 9V battery before handling wiring"],
        "learning_explanation": "Diodes only conduct in forward-bias condition.",
        "safety_warnings": ["Never short battery terminals directly."],
    }


# ==============================================================================
# 1. Text Input Validation Tests
# ==============================================================================

class TestTextInputValidation:
    """Tests for validating problem descriptions, circuit types, and components."""

    def test_empty_description_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_text_inputs("")

    def test_whitespace_only_description_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_text_inputs("   \n\t  ")

    def test_non_string_description_rejected(self):
        with pytest.raises(ValidationError, match="must be a string"):
            validate_text_inputs(12345)  # type: ignore

    def test_description_at_max_limit_accepted(self):
        valid_desc = "x" * MAX_DESCRIPTION_LENGTH
        validate_text_inputs(valid_desc)

    def test_description_exceeding_max_limit_rejected(self):
        oversized_desc = "x" * (MAX_DESCRIPTION_LENGTH + 1)
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validate_text_inputs(oversized_desc)

    def test_valid_optional_fields_accepted(self):
        validate_text_inputs(
            problem_description="LED not lighting up",
            circuit_type="Simple LED indicator",
            components="9V battery, 330 ohm resistor, red LED",
        )

    def test_circuit_type_exceeding_limit_rejected(self):
        oversized_type = "c" * (MAX_OPTIONAL_FIELD_LENGTH + 1)
        with pytest.raises(ValidationError, match="Circuit type exceeds maximum length"):
            validate_text_inputs(
                problem_description="LED not lighting up",
                circuit_type=oversized_type,
            )

    def test_components_exceeding_limit_rejected(self):
        oversized_components = "r" * (MAX_OPTIONAL_FIELD_LENGTH + 1)
        with pytest.raises(ValidationError, match="Component details exceed maximum length"):
            validate_text_inputs(
                problem_description="LED not lighting up",
                components=oversized_components,
            )

    def test_non_string_optional_fields_rejected(self):
        with pytest.raises(ValidationError, match="Circuit type must be a string"):
            validate_text_inputs(
                problem_description="Valid description",
                circuit_type=123,  # type: ignore
            )
        with pytest.raises(ValidationError, match="Component details must be a string"):
            validate_text_inputs(
                problem_description="Valid description",
                components=["resistor", "LED"],  # type: ignore
            )


# ==============================================================================
# 2. Image Validation Tests
# ==============================================================================

class TestImageValidation:
    """Tests for image format, size, and integrity validation."""

    def test_empty_image_bytes_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_image_bytes(b"")

    def test_non_bytes_image_rejected(self):
        with pytest.raises(ValidationError, match="must be provided as bytes"):
            validate_image_bytes("not-bytes")  # type: ignore

    def test_malformed_image_bytes_rejected(self):
        corrupted_bytes = b"GIF89a corrupted header and random invalid bytes"
        with pytest.raises(ValidationError, match="Invalid or corrupted"):
            validate_image_bytes(corrupted_bytes)

    def test_oversized_image_rejected(self):
        # 1 byte over the 10 MB limit
        oversized_bytes = b"0" * (MAX_IMAGE_SIZE_BYTES + 1)
        with pytest.raises(ValidationError, match="exceeds the maximum limit"):
            validate_image_bytes(oversized_bytes)

    def test_unsupported_image_formats_rejected(self):
        gif_bytes = create_test_image("GIF")
        with pytest.raises(ValidationError, match="Unsupported image format: 'GIF'"):
            validate_image_bytes(gif_bytes)

        bmp_bytes = create_test_image("BMP")
        with pytest.raises(ValidationError, match="Unsupported image format: 'BMP'"):
            validate_image_bytes(bmp_bytes)

    def test_valid_png_accepted(self, valid_png_bytes):
        fmt = validate_image_bytes(valid_png_bytes)
        assert fmt == "PNG"

    def test_valid_jpeg_accepted(self, valid_jpeg_bytes):
        fmt = validate_image_bytes(valid_jpeg_bytes)
        assert fmt == "JPEG"

    def test_valid_webp_accepted(self, valid_webp_bytes):
        fmt = validate_image_bytes(valid_webp_bytes)
        assert fmt == "WEBP"


# ==============================================================================
# 3. Prompt Construction Tests
# ==============================================================================

class TestPromptConstruction:
    """Tests for prompt templates and prompt builders."""

    def test_system_prompt_contains_core_requirements(self):
        assert "CircuitSense AI" in DIAGNOSTIC_SYSTEM_PROMPT
        assert "VISUAL EVIDENCE" in DIAGNOSTIC_SYSTEM_PROMPT
        assert "DISCONNECT POWER" in DIAGNOSTIC_SYSTEM_PROMPT
        assert "Low" in DIAGNOSTIC_SYSTEM_PROMPT and "High" in DIAGNOSTIC_SYSTEM_PROMPT
        assert "schema" in DIAGNOSTIC_SYSTEM_PROMPT.lower()

    def test_diagnostic_schema_has_required_keys(self):
        required_keys = DIAGNOSTIC_SCHEMA["required"]
        assert "summary" in required_keys
        assert "possible_faults" in required_keys
        assert "unknowns" in required_keys
        assert "safe_next_steps" in required_keys
        assert "learning_explanation" in required_keys
        assert "safety_warnings" in required_keys

    def test_build_user_prompt_with_description_only(self):
        prompt = build_user_prompt(problem_description="LED remains unlit")
        assert "### OBSERVED PROBLEM / SYMPTOMS:\nLED remains unlit" in prompt
        assert "CIRCUIT TYPE" not in prompt
        assert "KNOWN COMPONENTS" not in prompt
        assert "INSTRUCTIONS:" in prompt

    def test_build_user_prompt_with_all_context(self):
        prompt = build_user_prompt(
            problem_description="Motor does not spin",
            circuit_type="H-Bridge motor driver",
            components="L293D, 9V DC motor, 5V Arduino logic supply",
        )
        assert "Motor does not spin" in prompt
        assert "H-Bridge motor driver" in prompt
        assert "L293D, 9V DC motor" in prompt
        assert "INSTRUCTIONS:" in prompt

    def test_build_chat_payload_structure(self, valid_png_bytes):
        payload = build_chat_payload(
            model="gemma4:cloud",
            image_bytes=valid_png_bytes,
            problem_description="LED does not light",
            circuit_type="DC circuit",
            components="Resistor, LED",
        )
        assert payload["model"] == "gemma4:cloud"
        assert payload["stream"] is False
        assert payload["format"] == "json"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert len(payload["messages"][1]["images"]) == 1

    def test_system_prompt_discrepancy_handling(self):
        prompt = DIAGNOSTIC_SYSTEM_PROMPT
        prompt_lower = prompt.lower()
        # Must treat image as a snapshot and distinguish facts, reports, inferences
        assert "snapshot" in prompt_lower
        assert "inferred" in prompt_lower or "hypothesized" in prompt_lower
        # Must identify discrepancies between visual evidence and user description
        assert "discrepanc" in prompt_lower
        # Must forbid inventing explanations (such as timeline or delayed failure) without evidence
        assert "do not invent" in prompt_lower
        # Must emphasize confirming current circuit state as first action
        assert "confirming the current" in prompt_lower
        # Must allow empty possible_faults list when no fault is established
        assert "[]" in prompt
        # Must distinguish component verification from evidence of a fault
        assert "distinguish" in prompt_lower and "fault" in prompt_lower

    def test_system_prompt_electrical_measurement_safety(self):
        prompt = DIAGNOSTIC_SYSTEM_PROMPT
        prompt_lower = prompt.lower()
        # Must emphasize power disconnection and non-powered verification first
        assert "disconnect power" in prompt_lower
        assert "non-powered" in prompt_lower
        # Must forbid powered resistance measurements and direct current shorts
        assert "never instruct users to measure resistance on a powered circuit" in prompt_lower
        assert "never instruct users to measure current by placing multimeter probes directly across a power source" in prompt_lower
        # Must specify DC voltage mode and parallel connection for powered voltage checks
        assert "dc voltage" in prompt_lower
        assert "across" in prompt_lower
        # Must recommend instructor or qualified assistance
        assert "instructor" in prompt_lower or "qualified assistance" in prompt_lower

    def test_system_prompt_prohibits_unsupported_continuity_claims(self):
        prompt = DIAGNOSTIC_SYSTEM_PROMPT
        prompt_lower = prompt.lower()
        # Must describe placement and visible wires as visual observations, not proof of continuity
        assert "visual observations" in prompt_lower
        assert "never claim that a breadboard circuit forms a complete electrical path" in prompt_lower
        # Must distinguish between visually plausible layout and electrically verified connectivity
        assert "visually plausible circuit layout and electrically verified connectivity" in prompt_lower
        # Hidden contacts, rail breaks, internal connections must be treated as unknown
        assert "hidden" in prompt_lower and "unknown" in prompt_lower
        # Must not state circuit is functioning correctly merely because LED appears lit in photo
        assert "functioning correctly merely because an led appears illuminated" in prompt_lower
        # Must mandate qualified language when discussing circuit topology
        assert "appears to be arranged as" in prompt_lower

    def test_system_prompt_multimeter_safety_for_students(self):
        prompt = DIAGNOSTIC_SYSTEM_PROMPT
        prompt_lower = prompt.lower()
        # Prioritize markings, schematic, instructor
        assert "identifying component markings" in prompt_lower
        assert "schematic" in prompt_lower
        assert "instructor" in prompt_lower
        # Resistance requires power disconnect and isolating one lead
        assert "power source to be disconnected first" in prompt_lower
        assert "isolating or lifting one component lead" in prompt_lower or "lifting one resistor leg" in prompt_lower
        # Voltage measurements require DC voltage and safe parallel placement
        assert "dc voltage" in prompt_lower
        assert "across (in parallel with)" in prompt_lower
        # Prohibitions
        assert "measure resistance on a powered circuit" in prompt_lower
        assert "directly across a power source or battery" in prompt_lower or "directly across a battery or power supply" in prompt_lower
        # Practical and relevant advice without overwhelming students
        assert "concise" in prompt_lower or "intimidating" in prompt_lower

    def test_system_prompt_restricts_powered_voltage_measurements(self):
        prompt = DIAGNOSTIC_SYSTEM_PROMPT
        prompt_lower = prompt.lower()
        # Discourages unnecessary powered voltage measurements
        assert "do not recommend powered voltage measurements merely because component ratings or electrical values are unknown" in prompt_lower
        assert "no reported malfunction" in prompt_lower
        assert "materially help distinguish plausible explanations" in prompt_lower
        # Restricts powered measurements to competent users or appropriate supervision
        assert "someone competent to use a multimeter safely" in prompt_lower
        assert "instructor supervision" in prompt_lower
        # Prohibits asking inexperienced users to improvise probe placement
        assert "improvise probe placement" in prompt_lower
        # Prohibits inferring confirmed breadboard continuity from a photograph alone
        assert "apparent wire routing and component placement cannot prove internal breadboard continuity or a complete electrical path" in prompt_lower


# ==============================================================================
# 4. Model Response Parsing Tests
# ==============================================================================

class TestResponseParsing:
    """Tests for parsing raw assistant outputs into JSON dictionaries."""

    def test_valid_json_string_parsed(self, sample_diagnostic_dict):
        raw_json = json.dumps(sample_diagnostic_dict)
        parsed = parse_model_response(raw_json)
        assert parsed == sample_diagnostic_dict

    def test_markdown_code_fenced_json_parsed(self, sample_diagnostic_dict):
        raw_with_fences = f"```json\n{json.dumps(sample_diagnostic_dict, indent=2)}\n```"
        parsed = parse_model_response(raw_with_fences)
        assert parsed == sample_diagnostic_dict

    def test_empty_response_raises_error(self):
        with pytest.raises(OllamaResponseError, match="empty or non-string"):
            parse_model_response("")

    def test_malformed_json_raises_error(self):
        with pytest.raises(OllamaResponseError, match="Failed to parse model response as JSON"):
            parse_model_response("{malformed_json: true,")

    def test_non_dictionary_json_raises_error(self):
        with pytest.raises(OllamaResponseError, match="Expected a JSON object"):
            parse_model_response('["list", "of", "items"]')

        with pytest.raises(OllamaResponseError, match="Expected a JSON object"):
            parse_model_response('"just a plain string"')

    def test_empty_possible_faults_response_parsed(self):
        data = {
            "summary": "Photo shows illuminated LED while description reports unlit. Confirm state first.",
            "possible_faults": [],
            "unknowns": ["Current circuit state"],
            "safe_next_steps": ["Confirm if the circuit is currently energized"],
            "learning_explanation": "Visual snapshot at photograph time can differ from current state.",
            "safety_warnings": ["Disconnect power before changing wiring."],
        }
        parsed = parse_model_response(json.dumps(data))
        assert parsed["possible_faults"] == []
        assert isinstance(parsed["possible_faults"], list)


# ==============================================================================
# 5. Client diagnose_circuit Tests (Mocked HTTP)
# ==============================================================================

class TestDiagnoseCircuit:
    """Mocked unit tests for the main diagnose_circuit client entry point."""

    @patch("gemma_client.requests.post")
    def test_successful_diagnosis(
        self,
        mock_post,
        valid_png_bytes,
        sample_diagnostic_dict,
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "gemma4:cloud",
            "message": {
                "role": "assistant",
                "content": json.dumps(sample_diagnostic_dict),
            },
            "done": True,
        }
        mock_post.return_value = mock_resp

        result = diagnose_circuit(
            image_bytes=valid_png_bytes,
            problem_description="The LED is not lighting up.",
            circuit_type="Basic LED circuit",
            components="9V battery, 1k resistor, LED",
        )

        assert result == sample_diagnostic_dict
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/api/chat"
        assert kwargs["json"]["model"] == "gemma4:cloud"
        assert kwargs["timeout"] == 180

    @patch("gemma_client.requests.post")
    def test_environment_variable_override(
        self,
        mock_post,
        valid_png_bytes,
        sample_diagnostic_dict,
        monkeypatch,
    ):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-ollama:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "custom-gemma")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "custom-gemma",
            "message": {
                "role": "assistant",
                "content": json.dumps(sample_diagnostic_dict),
            },
        }
        mock_post.return_value = mock_resp

        result = diagnose_circuit(
            image_bytes=valid_png_bytes,
            problem_description="Testing custom env variables",
        )

        assert result == sample_diagnostic_dict
        args, kwargs = mock_post.call_args
        assert args[0] == "http://custom-ollama:11434/api/chat"
        assert kwargs["json"]["model"] == "custom-gemma"

    @patch("gemma_client.requests.post")
    def test_ollama_http_error(self, mock_post, valid_png_bytes):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        with pytest.raises(OllamaResponseError, match="HTTP error status 500"):
            diagnose_circuit(
                image_bytes=valid_png_bytes,
                problem_description="Testing HTTP 500",
            )

    @patch("gemma_client.requests.post")
    def test_ollama_timeout(self, mock_post, valid_png_bytes):
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        with pytest.raises(OllamaTimeoutError, match="timed out"):
            diagnose_circuit(
                image_bytes=valid_png_bytes,
                problem_description="Testing timeout handling",
            )

    @patch("gemma_client.requests.post")
    def test_ollama_connection_error(self, mock_post, valid_png_bytes):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(OllamaConnectionError, match="Failed to connect to Ollama"):
            diagnose_circuit(
                image_bytes=valid_png_bytes,
                problem_description="Testing connection failure",
            )

    @patch("gemma_client.requests.post")
    def test_ollama_reports_error_field(self, mock_post, valid_png_bytes):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"error": "model 'gemma4:cloud' not found"}
        mock_post.return_value = mock_resp

        with pytest.raises(OllamaResponseError, match="Ollama error reported"):
            diagnose_circuit(
                image_bytes=valid_png_bytes,
                problem_description="Testing Ollama error payload",
            )

    @patch("gemma_client.requests.post")
    def test_ollama_missing_content(self, mock_post, valid_png_bytes):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"role": "assistant"}}
        mock_post.return_value = mock_resp

        with pytest.raises(OllamaResponseError, match="missing expected 'message.content'"):
            diagnose_circuit(
                image_bytes=valid_png_bytes,
                problem_description="Testing missing message content",
            )

    def test_validation_fails_before_network_call(self):
        # Invalid input should fail immediately without any HTTP request
        with pytest.raises(ValidationError):
            diagnose_circuit(
                image_bytes=b"not-valid-image",
                problem_description="",
            )
