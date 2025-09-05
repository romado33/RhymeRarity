import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import parse_output

def test_parse_output_basic():
    text = (
        "Rhyme type: Perfect\n"
        "Rarity score: 10\n"
        "Explanation: clear example\n"
        "Examples: cat/hat"
    )
    expected = {
        "rhyme type": "Perfect",
        "rarity score": "10",
        "explanation": "clear example",
        "examples": "cat/hat",
    }
    assert parse_output(text) == expected

def test_parse_output_ignores_non_colon_and_handles_colons_in_value():
    text = (
        "Rhyme type: Perfect\n"
        "This line has no colon\n"
        "Explanation: Uses value: with colon\n"
    )
    expected = {
        "rhyme type": "Perfect",
        "explanation": "Uses value: with colon",
    }
    assert parse_output(text) == expected

def test_parse_output_empty_text_returns_empty_dict():
    assert parse_output("") == {}
