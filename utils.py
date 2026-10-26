"""
CodeSync — Utility functions for validation, JSON parsing, safety detection, and image handling.
"""

import json
import re
from typing import Dict, Any, Tuple, List, Optional
from PIL import Image
import io


def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Safely extracts and parses JSON from model output, handling markdown blocks,
    preceding/trailing commentary, and common formatting quirks.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Model returned an empty response.")

    text = raw_text.strip()

    # 1. Check for standard markdown code blocks ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # 2. Extract outermost curly braces if text still contains commentary
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]

    # 3. Attempt direct json.loads
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: remove trailing commas before closing braces/brackets
        cleaned = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse valid JSON from model output: {e}\nRaw output preview: {raw_text[:200]}...")


def normalize_confidence(val: Any, default: int = 70) -> int:
    """
    Normalizes confidence scores into an integer between 0 and 100.
    Handles floats like 0.85 -> 85, strings like '85%' -> 85, etc.
    """
    if val is None:
        return default
    try:
        if isinstance(val, (int, float)):
            if 0.0 <= val <= 1.0:
                return int(round(val * 100))
            return max(0, min(100, int(round(val))))
        
        # String case: e.g. "85%", "0.85", "85"
        val_str = str(val).strip().replace("%", "")
        float_val = float(val_str)
        if 0.0 <= float_val <= 1.0:
            return int(round(float_val * 100))
        return max(0, min(100, int(round(float_val))))
    except Exception:
        return default


def validate_circuit_input(symptoms: str) -> Tuple[bool, str]:
    """
    Validates user symptom input before initiating a model request.
    """
    if not symptoms or not symptoms.strip():
        return False, "Please describe what is wrong with your circuit before running the diagnosis."
    
    cleaned = symptoms.strip()
    if len(cleaned) < 8:
        return False, "Please provide a slightly more descriptive symptom (at least 8 characters) to allow accurate fault analysis."
    
    return True, ""


def validate_image_file(uploaded_file, max_size_mb: int = 10) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validates an uploaded circuit image or camera capture.
    Loads and copies image data safely to decouple from underlying streams.
    """
    if uploaded_file is None:
        return True, "", None
    
    # Check file size
    if hasattr(uploaded_file, "size") and uploaded_file.size > max_size_mb * 1024 * 1024:
        return False, f"Image file is too large ({uploaded_file.size / (1024 * 1024):.1f}MB). Maximum allowed size is {max_size_mb}MB.", None
    
    # Verify image integrity via PIL
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        allowed_formats = ["PNG", "JPEG", "JPG"]
        if image.format and image.format.upper() not in allowed_formats:
            return False, f"Unsupported image format: {image.format}. Please upload a PNG, JPG, or JPEG file.", None
        # Fully load pixel data into memory and copy so underlying stream isn't tied to display
        image.load()
        img_copy = image.copy()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return True, "", img_copy
    except Exception as e:
        return False, f"Invalid or corrupted image file: {str(e)}", None


def detect_safety_concerns(text: str) -> List[str]:
    """
    Scans input text for hazardous electrical indicators (mains, high voltage, etc.).
    """
    concerns = []
    text_lower = text.lower()
    
    mains_keywords = [
        "mains", "230v", "240v", "120v", "110v", "wall outlet", "wall plug",
        "ac mains", "high voltage", "grid power", "electric shock"
    ]
    for kw in mains_keywords:
        if kw in text_lower:
            concerns.append(
                "POTENTIAL MAINS VOLTAGE / AC HAZARD DETECTED: Disconnect wall power immediately before touching or probing. Use qualified supervision and isolated CAT-rated test equipment."
            )
            break

    thermal_keywords = ["smoking", "burning", "hot", "scorched", "melted", "fire"]
    for kw in thermal_keywords:
        if kw in text_lower:
            concerns.append(
                "THERMAL OVERHEATING WARNING: Turn off power immediately to prevent component rupture, fire, or skin burns."
            )
            break

    return concerns


def transcribe_audio_bytes(audio_bytes: bytes) -> Tuple[bool, str]:
    """
    Transcribes recorded audio bytes into text using SpeechRecognition.
    Uses Google's free speech recognition engine (no paid API key required).
    Returns (success, result_or_error_message).
    """
    if not audio_bytes or len(audio_bytes) < 44:
        return False, "Audio recording is empty."

    try:
        import speech_recognition as sr
    except ImportError:
        return False, "Speech recognition library is not installed in the environment."

    try:
        bio = io.BytesIO(audio_bytes)
        r = sr.Recognizer()
        with sr.AudioFile(bio) as source:
            audio_data = r.record(source)

        text = r.recognize_google(audio_data)
        if text and text.strip():
            return True, text.strip()
        return False, "No speech detected in recording."
    except sr.UnknownValueError:
        return False, "Could not detect clear speech in the recording. Please speak clearly into your microphone or type the symptoms manually."
    except sr.RequestError as e:
        return False, "Speech recognition service could not be reached. Please check network connectivity or type the circuit symptoms manually."
    except Exception as e:
        return False, f"Audio processing error: {str(e)}"

