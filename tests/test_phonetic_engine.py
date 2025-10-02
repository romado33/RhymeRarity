import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import PHONEMIZER_AVAILABLE, PhoneticEngine


def test_phonetic_engine_loads_cmudict():
    engine = PhoneticEngine()
    assert engine.cmudict, "Expected CMUdict to be loaded with entries"
    assert engine.get_phonemes("hat") == "HH AE1 T"


@pytest.mark.skipif(not PHONEMIZER_AVAILABLE, reason="phonemizer dependency is required")
def test_phonemizer_beats_naive_mapping_for_unseen_word():
    engine = PhoneticEngine()
    word = "chesterfieldian"

    naive_phonemes = engine._approximate_phonemes(word)
    phonemized = engine.get_phonemes(word)

    assert phonemized, "Phonemizer should return a phoneme sequence"
    assert phonemized != naive_phonemes
