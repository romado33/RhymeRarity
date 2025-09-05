"""Gradio application for analyzing rhyme strength and rarity.

This module defines a small web interface that sends two words to the OpenAI
API and classifies how well they rhyme.  The OpenAI key is read from the
environment at runtime, allowing the file to be imported without requiring the
key to be present immediately.
"""

from __future__ import annotations

import os
import textwrap
from collections import deque
import json
import atexit
from pathlib import Path

import gradio as gr
from openai import OpenAI


class MissingAPIKeyError(Exception):
    """Raised when the OpenAI API key cannot be found."""


class OpenAIClientError(Exception):
    """Raised when the OpenAI client cannot be initialized."""


class OpenAIRequestError(Exception):
    """Raised when a request to the OpenAI API fails."""

PROMPT_TEMPLATE = textwrap.dedent(
    """You are a rhyme expert.

    Evaluate the rhyme between the two inputs:
    Word 1: "{w1}"
    Word 2: "{w2}"

    Classify the rhyme using one of the following types:
    - Perfect (same ending vowel and consonant sounds, like cat/hat)
    - Slant (partial match in consonants/vowels, like worm/swarm or shape/keep)
    - Assonance (same vowel sounds only, like deep/green)
    - Consonance (same consonant sounds only, like blank/think)
    - Forced (sounds somewhat alike but not naturally rhyming, like orange/door hinge)
    - None (no rhyme at all)

    Then assess how rare this rhyme pairing is in published literature (0 = common, 100 = rare).

    Return the result in this exact format:
    Rhyme type: <type>
    Rarity score: <0–100>
    Explanation: <brief explanation>
    Examples: <optional>
    """
)


CACHE_FILE = Path(__file__).with_name("cache.json")
_CACHE: dict[tuple[str, str, str, float], str] = {}


def _load_cache() -> None:
    """Load cached results from disk."""

    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key, val in data.items():
                w1, w2, model, temp = key.split("|")
                _CACHE[(w1, w2, model, float(temp))] = val
        except Exception:  # pragma: no cover - cache read errors
            pass


def _save_cache() -> None:
    """Persist cached results to disk."""

    data = {
        "|".join([w1, w2, model, str(temp)]): val
        for (w1, w2, model, temp), val in _CACHE.items()
    }
    with CACHE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)


_load_cache()
atexit.register(_save_cache)


def _sanitize(word: str) -> str:
    """Strip surrounding whitespace and newlines from a word."""

    return word.strip().replace("\n", " ").replace("\r", " ")


def _get_openai_client() -> OpenAI:
    """Create an OpenAI client using the API key from the environment."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("OPENAI_API_KEY environment variable not set")
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:  # pragma: no cover - instantiation errors
        raise OpenAIClientError(str(e)) from e


def query_rhyme_score(
    w1: str, w2: str, *, model: str = "gpt-3.5-turbo", temperature: float = 0.7
) -> str:
    """Query the OpenAI API for rhyme analysis between two words.

    Results are cached based on the word pair, model, and temperature.
    """

    w1, w2 = _sanitize(w1), _sanitize(w2)
    if not w1 or not w2:
        return "Error: Both words must be provided."

    key = (w1, w2, model, float(temperature))
    if key in _CACHE:
        return _CACHE[key]

    prompt = PROMPT_TEMPLATE.format(w1=w1, w2=w2)

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            timeout=10,
        )
        content = response.choices[0].message.content
        _CACHE[key] = content
        return content
    except (MissingAPIKeyError, OpenAIClientError):
        raise
    except Exception as e:  # pragma: no cover - network errors
        raise OpenAIRequestError(str(e)) from e


def parse_output(text: str) -> dict[str, str]:
    """Parse the LLM response into a dictionary of results."""

    result: dict[str, str] = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            result[key.strip().lower()] = val.strip()
    return result


def analyze_rhyme(
    word1: str,
    word2: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
) -> tuple[str, str, str, str]:
    """Wrapper used by Gradio to return parsed rhyme information."""
    try:
        result = query_rhyme_score(word1, word2, model=model, temperature=temperature)
    except MissingAPIKeyError:
        msg = "OpenAI API key is missing. Set the OPENAI_API_KEY environment variable."
        return "", "", msg, ""
    except OpenAIClientError as e:
        msg = f"Failed to initialize OpenAI client: {e}"
        return "", "", msg, ""
    except OpenAIRequestError as e:
        msg = f"Network error or timeout contacting OpenAI: {e}"
        return "", "", msg, ""

    if result.startswith("Error"):
        return "", "", result, ""
    parsed = parse_output(result)
    return (
        parsed.get("rhyme type", "-"),
        parsed.get("rarity score", "-"),
        parsed.get("explanation", "-"),
        parsed.get("examples", "None"),
    )


def analyze_and_store(
    word1: str,
    word2: str,
    history: deque,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
) -> tuple[str, str, str, str, list[tuple[str, str, str]], deque]:
    """Analyze rhyme and update the history deque."""

    rhyme_type, rarity, explanation, examples = analyze_rhyme(
        word1, word2, model=model, temperature=temperature
    )
    if rhyme_type or rarity or explanation or examples:
        summary = f"{rhyme_type} ({rarity})"
        history.appendleft((word1, word2, summary))
    return rhyme_type, rarity, explanation, examples, list(history), history


def load_from_history(history: deque, evt: gr.SelectData) -> tuple[str, str]:
    """Return the word pair from the selected history row."""

    index = evt.index[0]
    try:
        word1, word2, _ = list(history)[index]
        return word1, word2
    except IndexError:  # pragma: no cover - UI safety
        return gr.update(), gr.update()


def clear_history() -> tuple[list[tuple[str, str, str]], deque]:
    """Clear the history state."""

    return [], deque(maxlen=5)


with gr.Blocks(title="🎤 Rhyme Rarity Checker") as demo:
    gr.Markdown("# 🎤 Rhyme Rarity Checker")
    gr.Markdown(
        "Enter two rhyming words and find out how well they rhyme, and how rare "
        "that rhyme pairing is. We consider all types of rhyme: perfect, slant, "
        "assonance, consonance, and forced rhymes."
    )
    history_state = gr.State(deque(maxlen=5))

    with gr.Row():
        with gr.Column():
            word1 = gr.Textbox(label="Word 1")
            word2 = gr.Textbox(label="Word 2")
            model_dd = gr.Dropdown(
                label="Model",
                choices=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4-turbo"],
                value="gpt-3.5-turbo",
            )
            temp_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            )
            submit = gr.Button("Analyze")
            rhyme_type = gr.Textbox(label="Rhyme Type")
            rarity = gr.Textbox(label="Rarity Score (0–100)")
            explanation = gr.Textbox(label="Explanation")
            examples = gr.Textbox(label="Examples")

        with gr.Column():
            gr.Markdown("### Recent History")
            history_table = gr.Dataframe(
                headers=["Word 1", "Word 2", "Result"],
                datatype=["str", "str", "str"],
                interactive=False,
            )
            clear_btn = gr.Button("Clear History")

    submit.click(
        analyze_and_store,
        inputs=[word1, word2, history_state, model_dd, temp_slider],
        outputs=[rhyme_type, rarity, explanation, examples, history_table, history_state],
    )
    clear_btn.click(clear_history, inputs=None, outputs=[history_table, history_state])
    history_table.select(load_from_history, inputs=[history_state], outputs=[word1, word2])


if __name__ == "__main__":  # pragma: no cover - UI entry point
    demo.launch()
