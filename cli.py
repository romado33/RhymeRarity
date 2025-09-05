import argparse
from typing import Sequence
from app import analyze_rhyme


def _pair_words(words: Sequence[str]) -> list[tuple[str, str]]:
    if len(words) % 2 != 0:
        raise ValueError("Please provide an even number of words representing pairs.")
    return [(words[i], words[i + 1]) for i in range(0, len(words), 2)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze rhyme type and rarity for word pairs."
    )
    parser.add_argument(
        "words",
        nargs="+",
        help="Words provided as pairs: word1 word2 [word3 word4 ...]",
    )
    args = parser.parse_args(argv)

    try:
        pairs = _pair_words(args.words)
    except ValueError as exc:
        parser.error(str(exc))

    header = f"{'Word 1':<15} {'Word 2':<15} {'Type':<12} {'Rarity':<6}"
    separator = f"{'-'*15} {'-'*15} {'-'*12} {'-'*6}"
    print(header)
    print(separator)

    for w1, w2 in pairs:
        rhyme_type, rarity, *_ = analyze_rhyme(w1, w2)
        print(f"{w1:<15} {w2:<15} {rhyme_type or '-':<12} {rarity or '-':<6}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
