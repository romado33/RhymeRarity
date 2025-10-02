import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import IntegratedRhymeGenerator


def test_integrated_rhyme_generator_latency():
    generator = IntegratedRhymeGenerator()

    # Warm up caches to avoid counting initialization overhead.
    generator.find_fully_integrated_rhymes(
        "hat",
        max_results=20,
        use_cultural_analysis=False,
    )

    start = time.perf_counter()
    results = generator.find_fully_integrated_rhymes(
        "hat",
        max_results=20,
        use_cultural_analysis=False,
    )
    duration = time.perf_counter() - start

    assert results, "Expected rhyme candidates to be returned for 'hat'"
    assert duration < 1.0, f"Integrated rhyme generation too slow: {duration:.3f}s"
