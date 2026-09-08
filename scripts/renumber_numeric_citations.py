#!/usr/bin/env python3
"""Renumber manuscript citations and references by first appearance."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT = ROOT / "paper" / "draft.md"
CITATION_PATTERN = re.compile(r"\[((?:\d+\s*(?:[-,]\s*)?)+)\]")
REFERENCE_PATTERN = re.compile(r"^(\d+)\.\s+(.+)$", re.MULTILINE)


def expand(group: str) -> list[int]:
    numbers: list[int] = []
    for part in group.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(value.strip()) for value in part.split("-", 1))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    return numbers


def collapse(numbers: list[int]) -> str:
    ordered = sorted(set(numbers))
    parts: list[str] = []
    start = ordered[0]
    end = ordered[0]
    for number in ordered[1:] + [ordered[-1] + 2]:
        if number == end + 1:
            end = number
            continue
        length = end - start + 1
        if length >= 3:
            parts.append(f"{start}-{end}")
        elif length == 2:
            parts.extend([str(start), str(end)])
        else:
            parts.append(str(start))
        start = end = number
    return ",".join(parts)


def renumber(manuscript: str) -> str:
    body, reference_block = manuscript.split("## References", 1)
    references = {
        int(number): text for number, text in REFERENCE_PATTERN.findall(reference_block)
    }
    first_appearance: list[int] = []
    for match in CITATION_PATTERN.finditer(body):
        for number in expand(match.group(1)):
            if number not in first_appearance:
                first_appearance.append(number)

    missing = sorted(set(first_appearance) - set(references))
    if missing:
        raise ValueError(f"Citations without numbered references: {missing}")

    mapping = {old: new for new, old in enumerate(first_appearance, start=1)}

    def replace_citation(match: re.Match[str]) -> str:
        return f"[{collapse([mapping[number] for number in expand(match.group(1))])}]"

    new_body = CITATION_PATTERN.sub(replace_citation, body).rstrip()
    new_references = [
        f"{mapping[old]}. {references[old]}" for old in first_appearance
    ]
    return f"{new_body}\n\n## References\n\n" + "\n".join(new_references) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    original = args.manuscript.read_text(encoding="utf-8")
    updated = renumber(original)
    if args.check:
        if updated != original:
            print("Numeric citations are not ordered by first appearance.")
            return 1
        print("Numeric citations are ordered by first appearance.")
        return 0

    args.manuscript.write_text(updated, encoding="utf-8")
    print(f"Renumbered {len(REFERENCE_PATTERN.findall(updated.split('## References', 1)[1]))} references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
