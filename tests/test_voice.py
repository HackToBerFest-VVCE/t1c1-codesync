"""
Unit and integration tests for CodeSync voice input and audio transcription.
"""

from unittest.mock import MagicMock, patch
import io
import wave
import struct
import pytest
from utils import transcribe_audio_bytes
from streamlit.testing.v1 import AppTest
from pathlib import Path

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def _create_dummy_wav_bytes() -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        data = struct.pack("<16000h", *([0] * 16000))
        wav.writeframes(data)
    return bio.getvalue()


def test_transcribe_audio_bytes_empty():
    success, msg = transcribe_audio_bytes(b"")
    assert not success
    assert "empty" in msg.lower()


def test_transcribe_audio_bytes_success():
    dummy_wav = _create_dummy_wav_bytes()
    with patch("speech_recognition.Recognizer.record") as mock_record, \
         patch("speech_recognition.Recognizer.recognize_google", return_value="The LED does not turn on at 5 volts"):
        success, text = transcribe_audio_bytes(dummy_wav)
        assert success
        assert text == "The LED does not turn on at 5 volts"


def test_transcribe_audio_bytes_unknown_value():
    import speech_recognition as sr
    dummy_wav = _create_dummy_wav_bytes()
    with patch("speech_recognition.Recognizer.record"), \
         patch("speech_recognition.Recognizer.recognize_google", side_effect=sr.UnknownValueError()):
        success, msg = transcribe_audio_bytes(dummy_wav)
        assert not success
        assert "speak clearly" in msg.lower()


def test_transcribe_audio_bytes_request_error():
    import speech_recognition as sr
    dummy_wav = _create_dummy_wav_bytes()
    with patch("speech_recognition.Recognizer.record"), \
         patch("speech_recognition.Recognizer.recognize_google", side_effect=sr.RequestError("offline")):
        success, msg = transcribe_audio_bytes(dummy_wav)
        assert not success
        assert "service could not be reached" in msg.lower()


def test_audio_input_populates_symptom_field():
    """Verify that uploading/recording audio via st.audio_input populates the symptom description."""
    dummy_wav = _create_dummy_wav_bytes()
    
    with patch("utils.transcribe_audio_bytes", return_value=(True, "5V rail is sagging to 1.8V")):
        at = AppTest.from_file(APP_PATH).run()
        # Find audio input
        audio_in = at.audio_input(key="circuit_voice_recorder")
        assert audio_in is not None
        
        # Simulate user recording audio
        audio_in.upload("voice_test.wav", dummy_wav, "audio/wav").run()
        
        # Verify transcription populated session state and symptoms field
        assert at.session_state["symptoms_text"] == "5V rail is sagging to 1.8V"
        assert at.text_area(key="symptoms_text_area").value == "5V rail is sagging to 1.8V"
