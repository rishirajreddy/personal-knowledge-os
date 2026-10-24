import sys
import os
import time
import threading
import queue
import whisper
import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

from llm import ask_llm_stream, clear_history
from tts import speak_stream

SAMPLE_RATE = 16000  # Native Whisper sampling rate
AUDIO_FILE = "recording.wav"

# Preload Whisper model
print("Loading Whisper model...", flush=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)
print(f"Whisper loaded on {device}", flush=True)


class PushToTalkRecorder:
    def __init__(self, samplerate=SAMPLE_RATE, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.recording = False
        self.audio_queue = queue.Queue()
        self.stream = None
        self.frames = []
    
    def callback(self, indata, frames, time_info, status):
        if status:
            pass
        if self.recording:
            self.frames.append(indata.copy())
    
    def start_recording(self):
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype='float32',
            callback=self.callback
        )
        self.stream.start()
    
    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.frames:
            audio_data = np.concatenate(self.frames, axis=0)
            return audio_data
        return np.array([])


recorder = PushToTalkRecorder()


def transcribe_audio_array(audio_data):
    if len(audio_data) == 0:
        return ""
    # Whisper expects float32 in range [-1, 1]
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    # Ensure mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    result = whisper_model.transcribe(audio_data, fp16=False)
    return result["text"].strip()


def main():
    print("\n=== Voice-to-Voice Assistant (Piper TTS) ===")
    print("Press Enter to start/stop recording. Type 'q' and Enter to quit.\n")
    
    try:
        while True:
            user_input = input("Press Enter to start recording (or 'q' to quit)... ").strip().lower()
            if user_input == 'q':
                print("Goodbye!")
                break
            
            print("Recording... Press Enter to stop.")
            recorder.start_recording()
            # Wait for user to press Enter to stop
            input()
            audio_data = recorder.stop_recording()
            
            if len(audio_data) == 0:
                print("No audio recorded. Try again.\n")
                continue
            
            # Check audio level
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 0.001:
                print("No speech detected (too quiet). Try again.\n")
                continue
            
            print("Transcribing...", flush=True)
            transcription = transcribe_audio_array(audio_data)
            if not transcription:
                print("No speech detected. Try again.\n")
                continue
            
            print(f"You: {transcription}\n")
            print("Assistant: ", end="", flush=True)
            
            # Stream response
            token_stream = ask_llm_stream(transcription)
            # Also print tokens as they come for console display
            def token_printer():
                pass
            
            # Use speak_stream which will handle sentence by sentence
            speak_stream(token_stream)
            print("\n")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except EOFError:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
