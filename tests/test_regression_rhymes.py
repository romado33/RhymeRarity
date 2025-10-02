import os
import sys

import pytest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import IntegratedRhymeGenerator


@pytest.fixture(scope="module")
def integrated_generator():
    # Disable cultural analysis to keep tests fast and deterministic.
    generator = IntegratedRhymeGenerator()
    return generator


@pytest.mark.parametrize("word", ["chesterfield", "window"])
def test_regression_rhymes_available(integrated_generator, word):
    results = integrated_generator.find_fully_integrated_rhymes(
        word,
        quality_threshold=10,  # allow near rhymes for regression coverage
        max_results=20,
        use_cultural_analysis=False,
    )

    assert results, f"Expected at least one rhyme candidate for '{word}'"
