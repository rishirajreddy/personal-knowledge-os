# Voice-to-Voice Assistant — Chat Context Export

## Project: personal-knowledge-os
Working directory: /Users/rishirajreddy/Development/AI-Workspace/personal-knowledge-os

## Summary
User has a Whisper-based voice-to-text setup connected to a local Qwen LLM (via Ollama). They want to convert it into a full voice-to-voice assistant with TTS output. Before executing, we planned the architecture.

## Current Codebase
- `main.py` — records 5s audio at 35 kHz, transcribes with Whisper base, calls `ask_llm`, prints result, calls `speak`.
- `llm.py` — `ask_llm(prompt)` posts to Ollama (qwen2.5-coder:7b, non-streaming).
- `tts.py` — `speak(text)` shells out to macOS `say`.
- `requirements.txt` — whisper, torch, sounddevice, scipy, requests.

## Plan Decisions (from user)
1. TTS engine: Local Neural TTS (Piper) — **Piper TTS** (user picked Piper over Kokoro).
2. Voice: `en_US-ryan-medium` (male).
3. Latency: **Sentence streaming** — speak sentences as LLM generates them.
4. Loop mode: **Push-to-Talk loop** — press Enter to record/stop, stays in conversation loop.
5. Formatting: **System prompt + cleaner** — voice system prompt + markdown stripper before TTS.
6. Whisper: pre-load model, use 16 kHz sampling, in-memory audio (no disk I/O).
7. LLM: streaming tokens from Ollama, multi-turn conversation history.

## Planned Architecture
- Record via sounddevice InputStream (push-to-talk).
- Transcribe in-memory with preloaded Whisper base.
- Stream LLM tokens from Ollama with `"stream": True`.
- Split on sentence boundaries, clean markdown, synthesize with Piper.
- Play audio via sounddevice.
- Loop until user quits.

## Plan File
Full plan saved at: /Users/rishirajreddy/.local/share/kilo/plans/1792818872886-voice-to-voice-tts-plan.md

## Next Steps (not yet executed)
- Update requirements.txt with piper-tts.
- Download Piper voice model (en_US-ryan-medium.onnx + json).
- Rewrite tts.py with Piper synthesizer + text cleaner + streaming player.
- Rewrite llm.py with streaming + voice system prompt + conversation history.
- Rewrite main.py with push-to-talk loop, preloaded Whisper, 16 kHz, in-memory audio.
- Add failure-mode handling (empty transcription, Ollama down, missing model, Ctrl+C).
