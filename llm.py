import json
import requests
from typing import Iterator, List, Dict

OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'qwen2.5-coder:7b'

VOICE_SYSTEM_PROMPT = """You are a helpful, conversational voice assistant. Keep your responses concise, direct, and conversational. Avoid code blocks, markdown tables, asterisks, and long bulleted lists unless explicitly asked. Speak naturally as if in conversation."""

conversation_history: List[Dict[str, str]] = []


def ask_llm_stream(prompt: str, use_history: bool = True, retrieved_memories: List[str] = None,
                   knowledge: List[str] = None,
                   notes_only: bool = False) -> Iterator[str]:
    messages = []
    system_content = VOICE_SYSTEM_PROMPT
    if retrieved_memories:
        memory_text = "\n".join(f"- {m}" for m in retrieved_memories)
        system_content += f"\n\nLong-term memories about the user:\n{memory_text}\nUse these facts when relevant."
    if knowledge or notes_only:
        snippets = "\n\n".join(knowledge) if knowledge else "(no matching excerpts found)"
        if notes_only:
            instruction = ("The user is asking about their own notes. Answer only from these excerpts and name "
                           "the note you used. If they don't contain the answer, say you couldn't find it in the notes.")
        else:
            instruction = ("These excerpts were found by keyword search and may be unrelated to the question. "
                           "Use them only if they clearly help; otherwise ignore them completely and answer "
                           "normally from your own knowledge. Never mention the excerpts or notes unless you used them.")
        system_content += f"\n\nExcerpts from the user's own notes:\n{snippets}\n{instruction}"
    messages.append({"role": "system", "content": system_content})
    if use_history and conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        
        full_response = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]
                    if content:
                        full_response.append(content)
                        yield content
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
        
        # Update history
        if use_history:
            conversation_history.append({"role": "user", "content": prompt})
            conversation_history.append({"role": "assistant", "content": "".join(full_response)})
            # Keep history manageable
            if len(conversation_history) > 20:
                conversation_history[:] = conversation_history[-20:]
    except Exception as e:
        error_msg = f"Ollama is not running or unreachable. Please start Ollama with ollama serve. Error: {e}"
        print(error_msg)
        yield error_msg


def ask_llm(prompt: str, retrieved_memories: List[str] = None):
    # Backward compatible non-streaming
    result = []
    for chunk in ask_llm_stream(prompt, retrieved_memories=retrieved_memories):
        result.append(chunk)
    return "".join(result)


def clear_history():
    global conversation_history
    conversation_history = []
