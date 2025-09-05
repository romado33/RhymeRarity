from __future__ import annotations
import argparse
from typing import List, Tuple, Sequence
from app import analyze_rhyme


def pair_words(words: Sequence[str]) -> List[Tuple[str, str]]:
    """Convert a sequence of words into pairs."""
    if len(words) % 2 != 0:
        raise ValueError("Please provide an even number of words representing pairs.")
    return [(words[i], words[i + 1]) for i in range(0, len(words), 2)]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze rhyme type and rarity for word pairs."
    )
    parser.add_argument(
        "words",
        nargs="+",
        help="Words provided as pairs: word1 word2 [word3 word4 ...]",
    )
    parser.add_argument(
        "--format",
        choices=["table", "simple"],
        default="table",
        help="Output format: table (aligned columns) or simple (fixed width)",
    )
    
    args = parser.parse_args(argv)
    
    try:
        pairs = pair_words(args.words)
    except ValueError as exc:
        parser.error(str(exc))
    
    # Collect results
    rows: List[Tuple[str, str, str, str]] = []
    for w1, w2 in pairs:
        rhyme_type, rarity, explanation, *_ = analyze_rhyme(w1, w2)
        if not rhyme_type and not rarity:
            rows.append((w1, w2, "Error", explanation or "Unknown error"))
        else:
            rows.append((w1, w2, rhyme_type or "-", str(rarity) if rarity else "-"))
    
    # Output results based on format choice
    if args.format == "table":
        # Dynamic column width formatting
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
    else:
        # Simple fixed-width formatting
        header = f"{'Word 1':<15} {'Word 2':<15} {'Type':<12} {'Rarity':<6}"
        separator = f"{'-'*15} {'-'*15} {'-'*12} {'-'*6}"
        
        print(header)
        print(separator)
        for w1, w2, rhyme_type, rarity in rows:
            print(f"{w1:<15} {w2:<15} {rhyme_type:<12} {rarity:<6}")
    
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())