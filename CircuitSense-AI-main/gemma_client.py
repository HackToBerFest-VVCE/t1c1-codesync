"""CircuitSense AI — Gemma 4 Ollama Vision Client.

This module provides a validated client function for sending circuit images
and diagnostic problem descriptions to Gemma 4 running on Ollama.
"""

import base64
import io
import json
import os
from typing import Any, Dict, Optional, Tuple

from PIL import Image
import requests

from prompts import DIAGNOSTIC_SYSTEM_PROMPT, build_user_prompt

# Default configuration
DEFAULT_OLLAMA_BASE_URL: str = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL: str = "gemma4:cloud"
DEFAULT_TIMEOUT_SECONDS: int = 180

# Validation constraints
MAX_DESCRIPTION_LENGTH: int = 2000
MAX_OPTIONAL_FIELD_LENGTH: int = 500
MAX_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
SUPPORTED_IMAGE_FORMATS: set[str] = {"PNG", "JPEG", "WEBP"}


class CircuitSenseError(Exception):
    """Base exception for all CircuitSense AI client operations."""


class ValidationError(CircuitSenseError, ValueError):
    """Raised when user inputs, parameters, or image bytes fail validation."""


class OllamaConnectionError(CircuitSenseError):
    """Raised when connecting to the Ollama endpoint fails."""


class OllamaTimeoutError(CircuitSenseError):
    """Raised when the Ollama request times out."""


class OllamaResponseError(CircuitSenseError):
    """Raised when Ollama returns an error status or malformed/missing response."""


def get_ollama_config() -> Tuple[str, str]:
    """Retrieve Ollama endpoint and model identifier from environment variables.

    Returns:
        Tuple of (base_url, model_name).
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    if not base_url:
        base_url = DEFAULT_OLLAMA_BASE_URL

    model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
    if not model:
        model = DEFAULT_OLLAMA_MODEL

    return base_url, model


def validate_text_inputs(
    problem_description: str,
    circuit_type: str = "",
    components: str = "",
) -> None:
    """Validate user text inputs for required content and length limits.

    Args:
        problem_description: Core description of observed circuit behavior.
        circuit_type: Optional description of intended circuit function.
        components: Optional list or details of known components.

    Raises:
        ValidationError: If any text input fails validation constraints.
    """
    if not isinstance(problem_description, str):
        raise ValidationError("Problem description must be a string.")

    cleaned_description = problem_description.strip()
    if not cleaned_description:
        raise ValidationError("Problem description cannot be empty or whitespace only.")

    if len(cleaned_description) > MAX_DESCRIPTION_LENGTH:
        raise ValidationError(
            f"Problem description exceeds maximum length of {MAX_DESCRIPTION_LENGTH} "
            f"characters (got {len(cleaned_description)})."
        )

    if circuit_type is not None:
        if not isinstance(circuit_type, str):
            raise ValidationError("Circuit type must be a string.")
        cleaned_circuit_type = circuit_type.strip()
        if len(cleaned_circuit_type) > MAX_OPTIONAL_FIELD_LENGTH:
            raise ValidationError(
                f"Circuit type exceeds maximum length of {MAX_OPTIONAL_FIELD_LENGTH} "
                f"characters (got {len(cleaned_circuit_type)})."
            )

    if components is not None:
        if not isinstance(components, str):
            raise ValidationError("Component details must be a string.")
        cleaned_components = components.strip()
        if len(cleaned_components) > MAX_OPTIONAL_FIELD_LENGTH:
            raise ValidationError(
                f"Component details exceed maximum length of {MAX_OPTIONAL_FIELD_LENGTH} "
                f"characters (got {len(cleaned_components)})."
            )


def validate_image_bytes(image_bytes: bytes) -> str:
    """Validate image bytes for presence, size, and supported image format.

    Args:
        image_bytes: Raw binary image data.

    Returns:
        The validated uppercase image format (e.g. 'PNG', 'JPEG', 'WEBP').

    Raises:
        ValidationError: If image data is empty, oversized, corrupt, or unsupported.
    """
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ValidationError("Image data must be provided as bytes or bytearray.")

    if len(image_bytes) == 0:
        raise ValidationError("Image bytes cannot be empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        max_mb = MAX_IMAGE_SIZE_BYTES // (1024 * 1024)
        raise ValidationError(f"Image size exceeds the maximum limit of {max_mb} MB.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
            fmt = (img.format or "").upper()
            if fmt == "JPG":
                fmt = "JPEG"
            if fmt not in SUPPORTED_IMAGE_FORMATS:
                allowed = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                raise ValidationError(
                    f"Unsupported image format: '{fmt}'. Allowed formats are: {allowed}."
                )
            return fmt
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Invalid or corrupted image data: {exc}") from exc


def build_chat_payload(
    model: str,
    image_bytes: bytes,
    problem_description: str,
    circuit_type: str = "",
    components: str = "",
) -> Dict[str, Any]:
    """Assemble the Ollama /api/chat JSON payload with base64 image and prompts.

    Args:
        model: Target model name.
        image_bytes: Validated image bytes.
        problem_description: Verified problem description.
        circuit_type: Optional verified circuit type.
        components: Optional verified component details.

    Returns:
        Dictionary ready to be sent as JSON to Ollama.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    user_content = build_user_prompt(
        problem_description=problem_description,
        circuit_type=circuit_type,
        components=components,
    )

    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": DIAGNOSTIC_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
                "images": [image_b64],
            },
        ],
        "stream": False,
        "format": "json",
    }


def parse_model_response(raw_text: str) -> Dict[str, Any]:
    """Clean and parse the raw string response from the model into a dictionary.

    Args:
        raw_text: The string content returned in the assistant message.

    Returns:
        Parsed dictionary representing the JSON object.

    Raises:
        OllamaResponseError: If the content is empty, invalid JSON, or not a dict.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise OllamaResponseError("Model returned empty or non-string response content.")

    cleaned = raw_text.strip()

    # Remove markdown code fence if wrapped by the model
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Find start and end of code block
        start_idx = 1
        end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        cleaned = "\n".join(lines[start_idx:end_idx]).strip()

    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        raise OllamaResponseError(f"Failed to parse model response as JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise OllamaResponseError(
            f"Expected a JSON object from model, but received {type(parsed).__name__}."
        )

    return parsed


def diagnose_circuit(
    image_bytes: bytes,
    problem_description: str,
    circuit_type: str = "",
    components: str = "",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Diagnose an electronic circuit fault from an image and problem description.

    Args:
        image_bytes: Binary data of the circuit image (PNG, JPEG, WebP, <= 10MB).
        problem_description: Description of the observed problem (1-2000 chars).
        circuit_type: Optional circuit category or expected behavior (<= 500 chars).
        components: Optional list of known components (<= 500 chars).
        timeout: Request timeout in seconds (defaults to 180s).

    Returns:
        Parsed diagnostic JSON dictionary conforming to DIAGNOSTIC_SCHEMA.

    Raises:
        ValidationError: If inputs or image fail validation.
        OllamaConnectionError: If connection to Ollama fails.
        OllamaTimeoutError: If the request exceeds the timeout.
        OllamaResponseError: If Ollama returns an error or invalid response.
        CircuitSenseError: For other unexpected operational errors.
    """
    # 1. Validate inputs
    validate_text_inputs(
        problem_description=problem_description,
        circuit_type=circuit_type,
        components=components,
    )
    validate_image_bytes(image_bytes=image_bytes)

    # 2. Get environment configuration
    base_url, model = get_ollama_config()

    # 3. Construct chat payload
    payload = build_chat_payload(
        model=model,
        image_bytes=image_bytes,
        problem_description=problem_description,
        circuit_type=circuit_type,
        components=components,
    )

    # 4. Dispatch request to Ollama
    endpoint_url = f"{base_url}/api/chat"

    try:
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise OllamaTimeoutError(
            f"Request to Ollama timed out after {timeout} seconds."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise OllamaConnectionError(
            f"Failed to connect to Ollama at {endpoint_url}. Verify Ollama is running."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise CircuitSenseError(f"HTTP request to Ollama failed: {exc}") from exc

    # 5. Check HTTP status code
    if response.status_code != 200:
        raise OllamaResponseError(
            f"Ollama returned HTTP error status {response.status_code}: {response.text}"
        )

    # 6. Parse envelope JSON
    try:
        result = response.json()
    except Exception as exc:
        raise OllamaResponseError(
            f"Failed to decode Ollama API response envelope as JSON: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise OllamaResponseError("Ollama API response envelope must be a JSON object.")

    if "error" in result and result["error"]:
        raise OllamaResponseError(f"Ollama error reported: {result['error']}")

    message = result.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise OllamaResponseError("Ollama response is missing expected 'message.content'.")

    # 7. Parse model content into structured dictionary
    return parse_model_response(message["content"])
