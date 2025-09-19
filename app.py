"""Core application logic for the Rhyme Rarity checker.

The original project split the user interface, OpenAI integration and
phonetic fallbacks across several modules.  For the Hugging Face space the
code was merged into a single file which unfortunately picked up a large
amount of duplicated, dead and even undefined code.  This rewrite keeps the
original public API used by the CLI and the tests while providing a clean and
well documented implementation.

Key features implemented here:
    * Sanitising helper used by the tests and CLI.
    * Parsing utilities to convert the LLM response into a dictionary.
    * Transparent caching so repeated queries do not re-hit the API.
    * Graceful fallback to the phonetic module when the API is unavailable.
    * A small Gradio interface mirroring the behaviour of the Hugging Face
      space.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, Iterable, Tuple

import gradio as gr
from openai import OpenAI, OpenAIError

from phonetic import fallback_rhyme

__all__ = [
    "CACHE_FILE",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "_sanitize",
    "parse_output",
    "query_rhyme_score",
    "analyze_rhyme",
    "create_interface",
    "_load_cache",
    "_save_cache",
]

# ---------------------------------------------------------------------------
# Configuration and cache handling
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("RHYME_RARITY_MODEL", "gpt-3.5-turbo")
DEFAULT_TEMPERATURE = float(os.getenv("RHYME_RARITY_TEMPERATURE", "0.7"))
CACHE_FILE = Path(os.getenv("RHYME_RARITY_CACHE", "cache.json"))

_CACHE: Dict[str, str] = {}
_CACHE_LOCK = threading.Lock()
_PROMPT_TEMPLATE = (
    "Analyse how well the following two words rhyme. Respond strictly with "
    "four lines in the format:\n"
    "Rhyme type: <type>\n"
    "Rarity score: <0-100>\n"
    "Explanation: <short explanation>\n"
    "Examples: <comma separated examples or '-'>\n\n"
    "Word 1: {word1}\n"
    "Word 2: {word2}"
)


def _sanitize(value: str) -> str:
    """Return a trimmed string with newlines collapsed to spaces."""

    collapsed = re.sub(r"[\r\n]+", " ", value)
    collapsed = re.sub(r"\s+", " ", collapsed)
    return collapsed.strip()


def parse_output(text: str) -> Dict[str, str]:
    """Parse the language model response into a case-insensitive dictionary.

    Only the first colon on each line is considered; additional colons are kept
    as part of the value.  Keys are normalised to lower case to make lookups
    straightforward for the CLI and tests.
    """

    parsed: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key:
            parsed[key] = value
    return parsed


def _cache_key(word1: str, word2: str, model: str, temperature: float) -> str:
    return f"{word1.lower()}|{word2.lower()}|{model}|{temperature}"


def _load_cache() -> None:
    """Populate the in-memory cache from ``CACHE_FILE`` if possible."""

    if _CACHE:
        # Already populated.  This keeps disk I/O to a minimum.
        return
    if not CACHE_FILE.exists():
        return
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if isinstance(data, dict):
        _CACHE.update({str(k): str(v) for k, v in data.items()})


def _save_cache() -> None:
    """Persist the in-memory cache to disk."""

    if not CACHE_FILE.parent.exists():
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(_CACHE, fh, indent=2, sort_keys=True)


def _get_openai_client() -> OpenAI:
    """Factory used for dependency injection in tests."""

    return OpenAI()


def _call_openai(
    client: OpenAI,
    word1: str,
    word2: str,
    *,
    model: str,
    temperature: float,
) -> str:
    """Invoke the Chat Completions API and return the raw message content."""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that classifies rhymes.",
        },
        {
            "role": "user",
            "content": _PROMPT_TEMPLATE.format(word1=word1, word2=word2),
        },
    ]
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    try:
        content = response.choices[0].message.content or ""
    except (AttributeError, IndexError):  # pragma: no cover - defensive guard
        return ""
    return content.strip()


def query_rhyme_score(
    word1: str,
    word2: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    client: OpenAI | None = None,
) -> str:
    """Return the raw analysis text for a pair of words.

    Results are cached using the pair of sanitised words, the model and the
    temperature.  If the OpenAI API call fails, the phonetic fallback is used
    instead.
    """

    clean1, clean2 = _sanitize(word1), _sanitize(word2)
    if not clean1 or not clean2:
        raise ValueError("Both words must contain visible characters.")

    _load_cache()
    cache_key = _cache_key(clean1, clean2, model, temperature)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
    if cached:
        return cached

    client = client or _get_openai_client()
    try:
        content = _call_openai(client, clean1, clean2, model=model, temperature=temperature)
    except OpenAIError:
        content = ""
    except Exception:  # pragma: no cover - extremely defensive
        content = ""

    if not content:
        content = fallback_rhyme(clean1, clean2)

    with _CACHE_LOCK:
        _CACHE[cache_key] = content
    return content


def analyze_rhyme(
    word1: str,
    word2: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    client: OpenAI | None = None,
) -> Tuple[str, str, str, str, str]:
    """Return structured rhyme information.

    The tuple contains ``(rhyme_type, rarity_score, explanation, examples,
    raw_response)``.  ``rarity_score`` is left as a string because the language
    model may return values such as ``"Unknown"``.
    """

    raw = query_rhyme_score(
        word1,
        word2,
        model=model,
        temperature=temperature,
        client=client,
    )
    parsed = parse_output(raw)
    return (
        parsed.get("rhyme type", ""),
        parsed.get("rarity score", ""),
        parsed.get("explanation", ""),
        parsed.get("examples", ""),
        raw,
    )


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def _format_gradio_result(result: Tuple[str, str, str, str, str]) -> Tuple[str, str, str, str, str]:
    rhyme_type, rarity, explanation, examples, raw = result
    return (
        rhyme_type or "-",
        rarity or "-",
        explanation or "No explanation returned.",
        examples or "-",
        raw,
    )


def create_interface() -> gr.Blocks:
    """Create the Gradio interface used by the Hugging Face space."""

    with gr.Blocks(title="Rhyme Rarity Checker") as demo:
        gr.Markdown(
            "# 🎤 Rhyme Rarity Checker\n"
            "Enter two words to see how strongly they rhyme and how rare the "
            "pairing is."
        )

        with gr.Row():
            word1 = gr.Textbox(label="Word 1", placeholder="e.g. cat")
            word2 = gr.Textbox(label="Word 2", placeholder="e.g. hat")
        analyze_button = gr.Button("Analyse", variant="primary")

        with gr.Row():
            rhyme_type = gr.Textbox(label="Rhyme type", interactive=False)
            rarity = gr.Textbox(label="Rarity score", interactive=False)
        explanation = gr.Textbox(label="Explanation", lines=4, interactive=False)
        examples = gr.Textbox(label="Examples", interactive=False)
        raw_output = gr.Textbox(label="Raw response", lines=6, interactive=False)

        def _run_analysis(w1: str, w2: str) -> Tuple[str, str, str, str, str]:
            try:
                result = analyze_rhyme(w1, w2)
            except ValueError as exc:
                return "", "", str(exc), "", ""
            return _format_gradio_result(result)

        analyze_button.click(
            _run_analysis,
            inputs=[word1, word2],
            outputs=[rhyme_type, rarity, explanation, examples, raw_output],
        )
        word2.submit(
            _run_analysis,
            inputs=[word1, word2],
            outputs=[rhyme_type, rarity, explanation, examples, raw_output],
        )

    return demo


def _main() -> None:  # pragma: no cover - manual invocation helper
    demo = create_interface()
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    _main()
