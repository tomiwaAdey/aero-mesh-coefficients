#!/usr/bin/env python3
"""Typeset the approved aerodynamic-mesh manuscript as a PDF."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SOURCE = PAPER / "draft.md"
METADATA = PAPER / "paper.yaml"
HEADER = PAPER / "header.tex"
IMAGE_FILTER = ROOT / "scripts" / "centre_paper_images.lua"
BUILD = ROOT / "tmp" / "paper"
OUTPUT = PAPER / "learning-aerodynamic-mesh-semantics.pdf"


def prepare_source() -> Path:
    """Move the approved title and abstract into Pandoc metadata."""
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    manuscript = SOURCE.read_text(encoding="utf-8")
    lines = manuscript.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("The manuscript must begin with one level-one title.")

    abstract_heading = lines.index("## Abstract")
    introduction_heading = next(
        index
        for index in range(abstract_heading + 1, len(lines))
        if lines[index].startswith("## 1.")
    )
    abstract_lines = lines[abstract_heading + 1 : introduction_heading]
    while abstract_lines and not abstract_lines[0].strip():
        abstract_lines.pop(0)
    while abstract_lines and not abstract_lines[-1].strip():
        abstract_lines.pop()

    metadata = METADATA.read_text(encoding="utf-8").rstrip()
    metadata += "\nabstract: |\n"
    metadata += "\n".join(f"  {line}" for line in abstract_lines)
    body = "\n".join(lines[introduction_heading:]).lstrip()
    body = re.sub(
        r"DOI: (10\.\S+?)\.(?=\s|$)",
        lambda match: f"DOI: \\nolinkurl{{{match.group(1)}}}.",
        body,
    )
    body = body.replace(
        "## References\n",
        "## References\n\n\\begingroup\n\\small\n\\RaggedRight\n",
        1,
    )
    body = f"{body.rstrip()}\n\n\\endgroup\n"

    prepared = BUILD / "manuscript.md"
    prepared.write_text(f"---\n{metadata}\n---\n\n{body}\n", encoding="utf-8")
    return prepared


def build_pdf() -> None:
    prepared = prepare_source()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "pandoc",
        str(prepared),
        "--from=markdown+tex_math_single_backslash-implicit_figures",
        "--standalone",
        "--pdf-engine=tectonic",
        f"--include-in-header={HEADER}",
        f"--lua-filter={IMAGE_FILTER}",
        f"--resource-path={ROOT}:{PAPER}:{BUILD}",
        "--variable=linestretch:1.06",
        "--variable=documentclass:article",
        "--output",
        str(OUTPUT),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
