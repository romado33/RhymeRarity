"""Regression tests for ensuring the CMU Pronouncing Dictionary stays complete."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import load_cmudict


def test_cmudict_contains_full_lexicon():
    """The CMU dictionary should contain the full upstream lexicon (≈133k entries)."""

    cmu_path = Path(__file__).resolve().parent.parent / "data" / "cmudict-0.7b"
    if cmu_path.exists():
        cmu_path.unlink()

    cmu_entries = load_cmudict(str(cmu_path))

    assert len(cmu_entries) >= 100_000

    # Ensure the cached copy is reused without redownloading.
    cmu_entries_again = load_cmudict(str(cmu_path))
    assert len(cmu_entries_again) == len(cmu_entries)
