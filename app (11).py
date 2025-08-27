# app.py (Gradio UI version for Hugging Face Spaces)

import openai
import os
import gradio as gr

# Load OpenAI API key from environment (set on Hugging Face Secrets)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to query OpenAI for rhyme evaluation
def query_rhyme_score(w1, w2):
    prompt = f"""
You're a rhyme evaluation expert.

Given the two words: "{w1}" and "{w2}", evaluate them based on the following:

1. **Rhyme Quality (0–100)** — how well they rhyme phonetically.  
2. **Rhyme Type** — perfect, slant, assonance, consonance, forced, or none.  
3. **Rarity Score (0–100)** — how rare this rhyme pairing is in published literature, music, or poetry (0 = very common, 100 = very rare).  
4. **Explanation** — brief reasoning covering rhyme strength and rarity.  
5. **Examples** — other word pairs with similar rhyme type/rarity, if any.

Format your reply exactly like this:

Rhyme Quality: <0–100>  
Rhyme Type: <one word>  
Rarity Score: <0–100>  
Explanation: <short paragraph>  
Examples: <comma-separated word pairs or 'None'>
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
        return result, "", "", "", ""
    parsed = parse_output(result)
    return (
        parsed.get("rhyme quality", "-"),
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
        gr.Textbox(label="Rhyme Quality (0–100)"),
        gr.Textbox(label="Rhyme Type"),
        gr.Textbox(label="Rarity Score (0–100)"),
        gr.Textbox(label="Explanation"),
        gr.Textbox(label="Examples")
    ],
    title="🎤 Rhyme Rarity Checker",
    description="Enter two rhyming words and find out how well they rhyme, and how rare that rhyme pairing is. We consider all types of rhyme: perfect, slant, assonance, consonance, and forced rhymes.",
    examples=[
        ["cat", "hat"],
        ["orange", "door hinge"],
        ["fire", "choir"],
        ["love", "above"]
    ]
)

demo.launch()
