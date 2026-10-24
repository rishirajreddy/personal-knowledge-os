import json
import requests
from typing import Iterator, List, Dict

OLLAMA_URL = 'http://localhost:11434/api/chat'
MODEL = 'qwen2.5-coder:7b'

VOICE_SYSTEM_PROMPT = """You are a helpful, conversational voice assistant. Keep your responses concise, direct, and conversational. Avoid code blocks, markdown tables, asterisks, and long bulleted lists unless explicitly asked. Speak naturally as if in conversation."""

conversation_history: List[Dict[str, str]] = []


def ask_llm_stream(prompt: str, use_history: bool = True) -> Iterator[str]:
    messages = []
    messages.append({"role": "system", "content": VOICE_SYSTEM_PROMPT})
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


def ask_llm(prompt: str):
    # Backward compatible non-streaming
    result = []
    for chunk in ask_llm_stream(prompt):
        result.append(chunk)
    return "".join(result)


def clear_history():
    global conversation_history
    conversation_history = []
