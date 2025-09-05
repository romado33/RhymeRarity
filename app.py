"""Gradio application for analyzing rhyme strength and rarity.

This module defines a small web interface that sends two words to the OpenAI
API and classifies how well they rhyme.  The OpenAI key is read from the
environment at runtime, allowing the file to be imported without requiring the
key to be present immediately.
"""

from __future__ import annotations

import os
import textwrap

import gradio as gr
from openai import OpenAI

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


def _sanitize(word: str) -> str:
    """Strip surrounding whitespace and newlines from a word."""

    return word.strip().replace("\n", " ").replace("\r", " ")


def _get_openai_client() -> OpenAI:
    """Create an OpenAI client using the API key from the environment."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def query_rhyme_score(w1: str, w2: str) -> str:
    """Query the OpenAI API for rhyme analysis between two words."""

    w1, w2 = _sanitize(w1), _sanitize(w2)
    if not w1 or not w2:
        return "Error: Both words must be provided."

    prompt = PROMPT_TEMPLATE.format(w1=w1, w2=w2)

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=10,
        )
        return response.choices[0].message.content
    except Exception as e:  # pragma: no cover - network errors
        return f"Error: {e}"


def parse_output(text: str) -> dict[str, str]:
    """Parse the LLM response into a dictionary of results."""

    result: dict[str, str] = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            result[key.strip().lower()] = val.strip()
    return result


def analyze_rhyme(word1: str, word2: str) -> tuple[str, str, str, str]:
    """Wrapper used by Gradio to return parsed rhyme information."""

    result = query_rhyme_score(word1, word2)
    if result.startswith("Error"):
        return "", "", result, ""
    parsed = parse_output(result)
    return (
        parsed.get("rhyme type", "-"),
        parsed.get("rarity score", "-"),
        parsed.get("explanation", "-"),
        parsed.get("examples", "None"),
    )


demo = gr.Interface(
    fn=analyze_rhyme,
    inputs=[
        gr.Textbox(label="Word 1"),
        gr.Textbox(label="Word 2"),
    ],
    outputs=[
        gr.Textbox(label="Rhyme Type"),
        gr.Textbox(label="Rarity Score (0–100)"),
        gr.Textbox(label="Explanation"),
        gr.Textbox(label="Examples"),
    ],
    title="🎤 Rhyme Rarity Checker",
    description=(
        "Enter two rhyming words and find out how well they rhyme, and how rare "
        "that rhyme pairing is. We consider all types of rhyme: perfect, slant, "
        "assonance, consonance, and forced rhymes."
    ),
)


if __name__ == "__main__":  # pragma: no cover - UI entry point
    demo.launch()
