from __future__ import annotations

import argparse
from typing import List, Tuple

from app import analyze_rhyme


def main(argv: list[str] | None = None) -> int:
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze rhyme type and rarity for multiple word pairs."
    )
    parser.add_argument(
        "words",
        nargs="+",
        help="Words supplied as pairs: word1 word2 [word3 word4 ...]",
    )
    args = parser.parse_args(argv)

    if len(args.words) % 2 != 0:
        parser.error("An even number of words is required.")

    pairs: List[Tuple[str, str]] = [
        (args.words[i], args.words[i + 1]) for i in range(0, len(args.words), 2)
    ]

    rows: List[Tuple[str, str, str, str]] = []
    for w1, w2 in pairs:
        rhyme_type, rarity, explanation, _ = analyze_rhyme(w1, w2)
        if not rhyme_type and not rarity:
            rows.append((w1, w2, "Error", explanation))
        else:
            rows.append((w1, w2, rhyme_type, rarity))

    headers = ["Word 1", "Word 2", "Rhyme Type", "Rarity"]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(str(col))) for i, col in enumerate(row)]

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in widths)
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(str(col).ljust(widths[i]) for i, col in enumerate(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
