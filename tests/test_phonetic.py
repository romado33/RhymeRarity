import os
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import phonetic  # noqa: E402
from app import parse_output  # noqa: E402


def test_load_cmudict_and_classify(tmp_path):
    data = """
;;; comment line
CAT  K AE1 T
HAT  HH AE1 T
""".strip()
    path = tmp_path / "cmudict.txt"
    path.write_text(data, encoding="utf-8")
    cmu = phonetic.load_cmudict(path)
    assert cmu["cat"] == [["K", "AE1", "T"]]
    assert phonetic.classify_pronunciations(cmu["cat"][0], cmu["hat"][0]) == "Perfect"


def test_fallback_rhyme_returns_expected_format():
    result = phonetic.fallback_rhyme("cat", "hat")
    parsed = parse_output(result)
    assert parsed["rhyme type"] == "Perfect"
    assert "rarity score" in parsed
