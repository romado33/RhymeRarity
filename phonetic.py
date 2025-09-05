"""Phonetic utilities for rhyme analysis.

This module provides simple helpers for loading the CMU Pronouncing
Dictionary and comparing phoneme sequences.  It is intentionally light
weight and only implements the small subset of functionality required for
unit tests and offline fallback rhyme analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# A very small built-in dictionary so the module works out of the box and in
# tests without requiring the full CMU dataset.  The :func:`load_cmudict` function
# can also read a real dictionary file if a path is provided.
_DEFAULT_DICT: Dict[str, List[List[str]]] = {
    "cat": [["K", "AE1", "T"]],
    "hat": [["HH", "AE1", "T"]],
    "dog": [["D", "AO1", "G"]],
    "log": [["L", "AO1", "G"]],
}


def load_cmudict(path: str | Path | None = None) -> Dict[str, List[List[str]]]:
    """Return a mapping of words to their phoneme pronunciations.

    Parameters
    ----------
    path:
        Optional path to a CMU Pronouncing Dictionary formatted text file.  If
        omitted, a tiny built-in dictionary is used.  Lines beginning with
        ``;;;`` are ignored and alternative pronunciations such as ``WORD(1)``
        are collapsed to the base word.
    """

    if path is None:
        # Return a copy so callers can freely mutate the result.
        return {k: [p[:] for p in v] for k, v in _DEFAULT_DICT.items()}

    cmu: Dict[str, List[List[str]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        word = parts[0].lower()
        if "(" in word:
            word = word[: word.index("(")]
        phones = parts[1:]
        cmu.setdefault(word, []).append(phones)
    return cmu


def _rhyming_part(pronunciation: List[str]) -> Tuple[str, ...]:
    """Return the phoneme slice used for rhyme comparison.

    The rhyming portion is defined as the segment starting at the last vowel
    phoneme (which in the CMU dictionary is denoted by a stress digit).
    """

    for i in range(len(pronunciation) - 1, -1, -1):
        phone = pronunciation[i]
        if phone[-1].isdigit():  # vowel phonemes include a stress digit
            return tuple(pronunciation[i:])
    return tuple(pronunciation)


def classify_pronunciations(p1: Iterable[str], p2: Iterable[str]) -> str:
    """Classify the rhyme type between two pronunciations.

    This is a very small heuristic implementation used for offline fallback.
    It distinguishes only a few rhyme categories and is **not** a full
    linguistic model.
    """

    r1 = _rhyming_part(list(p1))
    r2 = _rhyming_part(list(p2))
    if r1 == r2:
        return "Perfect"
    if r1 and r2 and r1[0] == r2[0] and r1[1:] != r2[1:]:
        return "Assonance"
    if r1[1:] == r2[1:] and r1[0] != r2[0]:
        return "Consonance"
    if r1[-1:] == r2[-1:] or r1[0] == r2[0]:
        return "Slant"
    return "None"


def _rarity_score(prons: List[List[str]], cmu: Dict[str, List[List[str]]]) -> int:
    """Estimate a rarity score based on rhyme population.

    The more words that share the same rhyming part, the lower the score.
    The score is a crude approximation: ``100 / count`` where ``count`` is the
    number of unique words with a matching rhyme part.
    """

    suffixes = {_rhyming_part(p) for p in prons}
    count = 0
    for word_prons in cmu.values():
        for p in word_prons:
            if _rhyming_part(p) in suffixes:
                count += 1
                break
    if count == 0:
        return 100
    score = int(100 / count)
    return max(0, min(100, score))


def fallback_rhyme(word1: str, word2: str, cmu: Dict[str, List[List[str]]] | None = None) -> str:
    """Return a rhyme analysis string compatible with ``query_rhyme_score``.

    This function is used as a fallback when the OpenAI API cannot be reached.
    It uses the CMU dictionary to provide a best-effort rhyme classification.
    """

    if cmu is None:
        cmu = load_cmudict()

    prons1 = cmu.get(word1.lower(), [])
    prons2 = cmu.get(word2.lower(), [])
    if not prons1 or not prons2:
        return (
            "Rhyme type: None\n"
            "Rarity score: 100\n"
            "Explanation: Word not found in pronunciation dictionary.\n"
            f"Examples: {word1}/{word2}"
        )

    types = {classify_pronunciations(p1, p2) for p1 in prons1 for p2 in prons2}
    priority = ["Perfect", "Slant", "Assonance", "Consonance", "None"]
    rhyme_type = next(t for t in priority if t in types)
    rarity = _rarity_score(prons1 + prons2, cmu)
    return (
        f"Rhyme type: {rhyme_type}\n"
        f"Rarity score: {rarity}\n"
        "Explanation: Phonetic fallback analysis.\n"
        f"Examples: {word1}/{word2}"
    )
