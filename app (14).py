# app.py (Gradio UI version for Hugging Face Spaces)

import openai
import os
import gradio as gr

# Load OpenAI API key from environment (set on Hugging Face Secrets)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to query OpenAI for rhyme evaluation
def query_rhyme_score(w1, w2):
    prompt = f"""
You are a rhyme expert.

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
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=10
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# Parse the output into dictionary
def parse_output(text):
    result = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            result[key.strip().lower()] = val.strip()
    return result

# Gradio interface function
def analyze_rhyme(word1, word2):
    result = query_rhyme_score(word1, word2)
    if result.startswith("Error"):
        return "", "", result, ""
    parsed = parse_output(result)
    return (
        parsed.get("rhyme type", "-"),
        parsed.get("rarity score", "-"),
        parsed.get("explanation", "-"),
        parsed.get("examples", "None")
    )

# Launch Gradio app
demo = gr.Interface(
    fn=analyze_rhyme,
    inputs=[
        gr.Textbox(label="Word 1"),
        gr.Textbox(label="Word 2")
    ],
    outputs=[
        gr.Textbox(label="Rhyme Type"),
        gr.Textbox(label="Rarity Score (0–100)"),
        gr.Textbox(label="Explanation")
    ],
    title="🎤 Rhyme Rarity Checker",
    description="Enter two rhyming words and find out how well they rhyme, and how rare that rhyme pairing is. We consider all types of rhyme: perfect, slant, assonance, consonance, and forced rhymes.",

)

demo.launch()
