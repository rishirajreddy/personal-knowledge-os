import sys
from pathlib import Path
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
from tts import speak_stream, speak
import memory
import knowledge
from listen import listen_for_utterance
from wake import WakeWordDetector, DEFAULT_WAKE_WORD, DEFAULT_THRESHOLD, chime

SAMPLE_RATE = 16000  # Native Whisper sampling rate
AUDIO_FILE = "recording.wav"

# Initialize memory DB
memory.init_db()

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


# Whisper tends to invent these on near-silent audio
HALLUCINATIONS = {"you", "thank you", "thanks for watching", "bye"}
QUIT_PHRASES = {"quit", "exit", "goodbye", "good bye", "stop listening", "bye"}


FOLLOW_UP_SEC = 6      # after an answer, keep listening this long without the wake word
WAKE_TIMEOUT_SEC = 8   # after the wake word, wait this long for you to start speaking


def capture_audio(push_to_talk: bool, detector=None, in_conversation: bool = False):
    """Return recorded audio (empty if nothing was said), or None if the user asked to quit."""
    if push_to_talk:
        user_input = input("Press Enter to start recording (or 'q' to quit)... ").strip().lower()
        if user_input == 'q':
            return None
        print("Recording... Press Enter to stop.")
        recorder.start_recording()
        input()
        return recorder.stop_recording()
    if detector and not in_conversation:
        print(f"Waiting for '{detector.wake_word.replace('_', ' ')}'...", flush=True)
        detector.wait()
        chime()
        print("Listening...", flush=True)
        return listen_for_utterance(wait_timeout=WAKE_TIMEOUT_SEC)
    print("Listening...", flush=True)
    return listen_for_utterance(wait_timeout=FOLLOW_UP_SEC if detector else None)


def arg_value(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv[:-1] else default


def main():
    push_to_talk = "--push-to-talk" in sys.argv
    use_wake_word = not push_to_talk and "--no-wake-word" not in sys.argv
    detector = None
    if use_wake_word:
        print("Loading wake-word model...", flush=True)
        detector = WakeWordDetector(arg_value("--wake-word", DEFAULT_WAKE_WORD),
                                    float(arg_value("--wake-threshold", str(DEFAULT_THRESHOLD))))
    print("\n=== Voice-to-Voice Assistant (Piper TTS) ===")
    if push_to_talk:
        print("Push-to-talk: press Enter to start/stop recording, 'q' to quit.")
    elif detector:
        print(f"Say '{detector.wake_word.replace('_', ' ')}' then ask. Follow-ups within {FOLLOW_UP_SEC}s need no wake word. Say 'goodbye' or press Ctrl-C to quit.")
    else:
        print("Always listening: just speak; I'll respond when you pause. Say 'goodbye' or press Ctrl-C to quit.")
    print("Memory: 'Remember that ...', 'What do you remember?', 'Forget that ...'.\n")
    
    in_conversation = False
    try:
        while True:
            audio_data = capture_audio(push_to_talk, detector, in_conversation)
            if audio_data is None:
                print("Goodbye!")
                break
            
            if len(audio_data) == 0:
                in_conversation = False  # nobody spoke: go back to waiting for the wake word
                if push_to_talk:
                    print("No audio recorded. Try again.\n")
                continue
            
            # Check audio level
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 0.001:
                print("[too quiet - ignored]", flush=True)
                continue
            
            print("Transcribing...", flush=True)
            transcription = transcribe_audio_array(audio_data)
            if not transcription:
                print("[no words recognised]", flush=True)
                continue
            if not push_to_talk and transcription.lower().strip(" .,!?") in HALLUCINATIONS - QUIT_PHRASES:
                print(f"[ignored likely noise: {transcription!r}]", flush=True)
                continue
            
            print(f"You: {transcription}\n")
            if transcription.lower().strip(" .,!?'\"") in QUIT_PHRASES:
                speak("Goodbye!")
                break
            
            # Explicit memory commands: remember / what do you remember / forget
            reply = memory.handle_memory_command(transcription) or knowledge.handle_knowledge_command(transcription)
            if reply:
                print(f"Assistant: {reply}")
                speak(reply)
                print()
                in_conversation = True
                continue
            
            # Retrieve relevant memories
            relevant = memory.get_relevant_memories(transcription, limit=5)
            memory_ids = [m['id'] for m in relevant]
            memory_facts = [m['fact'] for m in relevant]
            if memory_facts:
                print(f"[Memory] Recalling {len(memory_facts)} fact(s)...")
            
            notes_only = knowledge.asks_about_notes(transcription)
            notes = knowledge.search(transcription, explicit=notes_only)
            if notes:
                print(f"[Notes] Using {len(notes)} excerpt(s) from: {', '.join(sorted({Path(n['path']).name for n in notes}))}")
            
            print("Assistant: ", end="", flush=True)
            
            # Stream response with memories
            token_stream = ask_llm_stream(transcription, retrieved_memories=memory_facts if memory_facts else None,
                                         knowledge=knowledge.format_snippets(notes) or None,
                                         notes_only=notes_only)
            speak_stream(token_stream)
            in_conversation = True
            
            # Mark memories as used
            if memory_ids:
                memory.mark_used(memory_ids)
            
            print("\n")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except EOFError:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
