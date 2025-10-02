import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import PhoneticEngine


def test_phonetic_engine_loads_cmudict():
    engine = PhoneticEngine()
    assert engine.cmudict, "Expected CMUdict to be loaded with entries"
    assert engine.get_phonemes("hat") == "HH AE1 T"
