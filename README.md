# Personal Knowledge OS

A local-first voice assistant for macOS. Talk to it, and it answers aloud. It remembers things you tell it, and it can answer questions from your own notes. Speech recognition, the language model, text-to-speech and memory all run on your machine, so nothing personal is sent to a cloud service.

| Part | Technology |
| --- | --- |
| Speech to text | [OpenAI Whisper](https://github.com/openai/whisper) (`base` model, runs locally) |
| Language model | [Ollama](https://ollama.com) running `qwen2.5-coder:7b` |
| Text to speech | [Piper](https://github.com/rhasspy/piper) (`en_US-ryan-medium`) |
| Wake word | [openWakeWord](https://github.com/dscripka/openWakeWord) (`alexa` by default) |
| Memory and notes | SQLite (plain tables for memory, FTS5 full-text search for notes) |

## Features

- **Hands-free conversation.** Say the wake word, then ask. It detects when you stop speaking, replies aloud, and keeps listening for follow-ups for a few seconds without needing the wake word again.
- **Persistent memory.** It remembers facts across restarts, and only pulls relevant ones into each prompt.
- **Notes search.** Point it at folders of `.md`, `.txt` and `.rst` files and ask about them in conversation.
- **Conversation context.** Follow-up questions work without repeating yourself.

## Requirements

- macOS on Apple Silicon (developed on a MacBook Air with 16 GB RAM). The chime uses `afplay`.
- Python 3 (developed on 3.14).
- [Ollama](https://ollama.com) installed and running.
- A microphone, and permission for your terminal to use it (System Settings, Privacy & Security, Microphone).

## Setup

```bash
git clone <this repo>
cd personal-knowledge-os

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Language model
ollama pull qwen2.5-coder:7b

# Piper voice (about 63 MB, kept out of git)
mkdir -p voices
curl -L -o voices/en_US-ryan-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx
curl -L -o voices/en_US-ryan-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json
```

Check that the voice downloaded completely: `en_US-ryan-medium.onnx` should be 63,201,294 bytes. A truncated file fails to load, and the assistant then silently falls back to the macOS `say` voice (it prints a warning).

The Whisper and wake-word models download automatically on first run.

## Run

```bash
ollama serve            # if Ollama is not already running
python main.py
```

Say "Alexa", wait for the chime, then ask your question. Say "goodbye" (or press Ctrl-C) to quit.

| Option | Effect |
| --- | --- |
| `--wake-word NAME` | `alexa` (default), `hey_jarvis`, `hey_mycroft` or `hey_rhasspy` |
| `--wake-threshold 0.6` | Raise if it wakes by accident, lower if it misses you (default 0.5) |
| `--no-wake-word` | Always listening, no wake word |
| `--push-to-talk` | Press Enter to start and stop recording |

## Memory

Memory is stored in `data/memories.db` and is never loaded wholesale into a prompt. Only memories that share meaningful words with your question are included.

- "Remember that my project is called Atlas."
- "What do you remember?"
- "Forget that my project is called Atlas." (If several memories match equally well, it deletes nothing and asks you to be more specific.)

Not everything you say becomes a memory. Only things you explicitly ask it to remember are saved.

## Notes

```bash
python knowledge.py add ~/Notes     # register a folder (repeat for more)
python knowledge.py index           # read new or changed files
python knowledge.py sources         # list registered folders
python knowledge.py remove ~/Notes
python knowledge.py search "atlas deployment"   # see what it would find
```

After editing notes, say "reindex my notes". Files are split into chunks of about 800 characters, indexed with SQLite FTS5, and re-read only when they change. Hidden files and folders are skipped.

- For ordinary questions, a note is offered to the model only when at least two distinctive words from your question appear in the same chunk. The model is told to ignore it if it's unrelated, so general questions get normal answers.
- If you say "my notes", "my documents" and so on, it answers strictly from your notes and says when it can't find the answer.

Search is keyword-based, not semantic. If you ask using different words than your notes use, it can miss. Semantic search is a possible future upgrade.

## Privacy

Everything runs locally. Memories and the notes index live in `data/`, and your note folders are never copied. Both are listed in `.gitignore`, as is the `notes/` folder.

## Project layout

```
main.py        Voice loop: listen, transcribe, route, respond
listen.py      Records one utterance and detects when you stop speaking
wake.py        Wake-word detection
llm.py         Ollama chat with history, memories and note excerpts
tts.py         Piper text to speech
memory.py      Persistent memory and memory voice commands
knowledge.py   Notes indexing and search
```

## Roadmap

- Interrupting the assistant while it is speaking (an early attempt was removed; it needs more work on echo from laptop speakers).
- Computer actions (open apps, create folders, run approved scripts) with confirmation before anything destructive.
- Optional semantic search for notes if keyword search proves insufficient.

## Known limitations

- Voice interaction is English only.
- macOS only.
- A reply cannot be interrupted once it starts.

## License

[MIT](LICENSE)
