import os
import dotenv
import gradio as gr
from openai import OpenAI
import requests
import json
from typing import Generator, List, Dict, Optional
import re


ollama_api = "http://localhost:11434/api/chat"
headers = {"Content-type":"application/json"}
model = "llama2:7b"

system_prompt = "You are helpful assistant who helps freelancer to get clients."

_json_obj_re = re.compile(rb"\{(?:[^{}]|(?R))*\}", re.DOTALL)  # binary-safe recursive regex (works in Python's re)

def _safe_decode_line(raw_line) -> str:
    if raw_line is None:
        return ""
    if isinstance(raw_line, bytes):
        try:
            return raw_line.decode("utf-8", errors="ignore").strip()
        except Exception:
            return raw_line.decode("latin-1", errors="ignore").strip()
    return str(raw_line).strip()

def _extract_text_from_parsed(parsed: dict) -> Optional[str]:
    """
    Extract a text fragment from Ollama's JSON shapes.
    """
    if not isinstance(parsed, dict):
        return None
    # Ollama chunk: {"message": {"role":"assistant","content":"..."}}
    msg = parsed.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str) and content:
            return content
        # sometimes content might be list/parts
        if isinstance(content, list):
            return "".join([p for p in content if isinstance(p, str)])
    # fallback: top-level 'content' or 'text'
    if isinstance(parsed.get("content"), str) and parsed.get("content"):
        return parsed.get("content")
    if isinstance(parsed.get("text"), str) and parsed.get("text"):
        return parsed.get("text")
    return None

def chat_with_ollama_stream(message: str, history: List[Dict], stream: bool = True) -> Generator[str, None, None]:
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    payload = {"model": model, "messages": messages, "stream": stream}

    with requests.post(ollama_api, json=payload, stream=True, headers={"Content-Type": "application/json"}) as r:
        r.raise_for_status()
        response = ""

        for raw_line in r.iter_lines(decode_unicode=False):
            line = _safe_decode_line(raw_line)
            if not line:
                continue

            if line.startswith("data:"):
                line = line[len("data:"):].strip()

            if line == "[DONE]":
                break

            # Try parse the whole line as JSON
            parsed = None
            text_piece = None
            try:
                parsed = json.loads(line)
                text_piece = _extract_text_from_parsed(parsed)
            except Exception:
                parsed = None

            # If that didn't work, try to find multiple JSON objects concatenated on the same line
            if text_piece is None:
                try:
                    # work on bytes to use the recursive regex safely
                    raw_bytes = raw_line if isinstance(raw_line, bytes) else line.encode("utf-8", errors="ignore")
                    matches = list(_json_obj_re.finditer(raw_bytes))
                    if matches:
                        for m in matches:
                            try:
                                chunk = json.loads(m.group(0))
                                piece = _extract_text_from_parsed(chunk)
                                if piece:
                                    response += piece
                                    yield response
                            except Exception:
                                continue
                        # continue outer loop (we already yielded inside)
                        continue
                except Exception:
                    # ignore regex failures and fall back to plain-text append
                    pass

            # If we got a text_piece from the single-JSON attempt, append it
            if text_piece:
                response += text_piece
                yield response
                continue

            # If nothing matched as JSON, treat the line as plain text and append
            response += line
            yield response

        if response:
            yield response

gr.ChatInterface(fn=chat_with_ollama_stream,
                 title="Adversarial Chatbot for Freelancers",
                 description="A chatbot that helps freelancers to get clients by providing tailored advice and strategies.",
                 theme="soft",
                 ).launch()