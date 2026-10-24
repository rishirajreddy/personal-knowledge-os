import os
import re
import wave
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd

# Try to import piper
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except Exception:
    PIPER_AVAILABLE = False

VOICE_DIR = Path(__file__).parent / "voices"
ONNX_PATH = VOICE_DIR / "en_US-ryan-medium.onnx"
JSON_PATH = VOICE_DIR / "en_US-ryan-medium.onnx.json"

_voice = None


def _ensure_voice():
    global _voice
    if _voice is not None:
        return _voice
    if not PIPER_AVAILABLE:
        raise RuntimeError("piper-tts not available")
    if not ONNX_PATH.exists() or not JSON_PATH.exists():
        raise FileNotFoundError(
            f"Missing Piper voice files in {VOICE_DIR}. "
            f"Need {ONNX_PATH.name} and {JSON_PATH.name}"
        )
    _voice = PiperVoice.load(str(ONNX_PATH), config_path=str(JSON_PATH))
    return _voice


def clean_text_for_speech(text: str) -> str:
    if not text:
        return ""
    # Remove code blocks
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`[^`]+`', ' ', text)
    # Remove markdown images/links (keep text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def speak_sentence(sentence: str) -> None:
    sentence = clean_text_for_speech(sentence)
    if not sentence:
        return
    try:
        voice = _ensure_voice()
        # PiperVoice.synthesize returns audio chunks
        audio_stream = voice.synthesize(sentence)
        # Collect all audio
        all_data = []
        sample_rate = 22050  # default for piper; will be updated if available
        for chunk in audio_stream:
            # chunk may have sample_rate and audio
            if hasattr(chunk, 'audio_float_array'):
                audio = chunk.audio_float_array
                if hasattr(chunk, 'sample_rate'):
                    sample_rate = chunk.sample_rate
                all_data.append(audio)
            elif isinstance(chunk, tuple) and len(chunk) == 2:
                sr, audio = chunk
                sample_rate = sr
                all_data.append(audio)
            elif hasattr(chunk, 'samples'):
                audio = chunk.samples
                if hasattr(chunk, 'sample_rate'):
                    sample_rate = chunk.sample_rate
                all_data.append(audio)
            else:
                # try to access common attrs
                try:
                    audio = np.array(chunk.audio)
                    if hasattr(chunk, 'rate'):
                        sample_rate = chunk.rate
                    all_data.append(audio)
                except Exception:
                    pass
        if not all_data:
            return
        audio = np.concatenate(all_data)
        # Normalize to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        # Fallback to macOS say
        try:
            import subprocess
            subprocess.run(["say", sentence])
        except Exception:
            print(f"[TTS Error] {e}")


def speak_stream(token_generator: Iterator[str]) -> None:
    buffer = ""
    # sentence terminators
    terminators = ['.', '?', '!', '\n']
    for token in token_generator:
        buffer += token
        # Check for sentence boundary
        # Look for terminator followed by space or end
        for i, ch in enumerate(buffer):
            if ch in terminators:
                # Extract sentence up to and including terminator
                sentence = buffer[:i + 1].strip()
                if sentence:
                    speak_sentence(sentence)
                buffer = buffer[i + 1:]
                break
        else:
            continue  # executed when no break in for
        # Reset loop over buffer
    # Flush remaining
    if buffer.strip():
        speak_sentence(buffer.strip())


def speak(text: str) -> None:
    # For backward compatibility
    speak_sentence(text)
