#!/usr/bin/env python3
"""Validate and compile the GeoCebada technical report.

This builder intentionally DOES NOT regenerate scientific figures or analyses.
It treats the versioned report figures under reports/ as frozen inputs and only:
1. validates LaTeX inputs, figure references, and citation keys;
2. compiles the report with the repository's documented latexmk command.

Usage
-----
python reporte/tecnico/build.py --check
python reporte/tecnico/build.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[1]
MAIN = REPORT_DIR / "main.tex"
BIB = REPORT_DIR / "references.bib"

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
FIGURE_RE = re.compile(r"\\safefigure(?:\[[^\]]+\])?\{([^}]+)\}")
CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|parencite|textcite)\{([^}]+)\}"
)
BIBKEY_RE = re.compile(r"@\w+\{([^,]+),")


def _tex_files() -> list[Path]:
    return sorted(REPORT_DIR.rglob("*.tex"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def validate() -> None:
    if not MAIN.is_file():
        raise FileNotFoundError(f"Missing main TeX file: {MAIN}")
    if not BIB.is_file():
        raise FileNotFoundError(f"Missing bibliography: {BIB}")
    if not (REPORT_DIR / "velvetblue.sty").is_file():
        raise FileNotFoundError("Missing report style: reporte/tecnico/velvetblue.sty")

    main_text = _read(MAIN)
    missing_inputs: list[str] = []
    for rel in INPUT_RE.findall(main_text):
        path = REPORT_DIR / f"{rel}.tex"
        if not path.is_file():
            missing_inputs.append(str(path.relative_to(ROOT)))

    missing_figures: list[str] = []
    cited_keys: set[str] = set()
    for tex_path in _tex_files():
        text = _read(tex_path)

        for rel in FIGURE_RE.findall(text):
            figure = (REPORT_DIR / rel).resolve()
            if not figure.is_file():
                try:
                    shown = figure.relative_to(ROOT)
                except ValueError:
                    shown = figure
                missing_figures.append(str(shown))

        for group in CITE_RE.findall(text):
            cited_keys.update(key.strip() for key in group.split(",") if key.strip())

    bib_keys = set(BIBKEY_RE.findall(_read(BIB)))
    missing_citations = sorted(cited_keys - bib_keys)

    problems: list[str] = []
    if missing_inputs:
        problems.append("Missing LaTeX inputs:\n  - " + "\n  - ".join(missing_inputs))
    if missing_figures:
        problems.append(
            "Missing frozen figure inputs:\n  - "
            + "\n  - ".join(sorted(set(missing_figures)))
        )
    if missing_citations:
        problems.append(
            "Citation keys missing from references.bib:\n  - "
            + "\n  - ".join(missing_citations)
        )

    if problems:
        raise RuntimeError("\n\n".join(problems))

    print(
        "Technical report validation: PASS "
        f"({len(_tex_files())} TeX files, "
        f"{len(cited_keys)} cited bibliography keys)."
    )
    print("Figures are treated as frozen inputs; no scientific runner was executed.")


def compile_report() -> None:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError(
            "latexmk is not available. Install a LaTeX distribution with latexmk and Biber."
        )

    subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "main.tex"],
        cwd=REPORT_DIR,
        check=True,
    )

    pdf = REPORT_DIR / "main.pdf"
    if not pdf.is_file():
        raise RuntimeError("latexmk finished but main.pdf was not produced.")
    print(f"Technical report build: PASS -> {pdf.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs, figures, and citation keys without compiling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate()
        if not args.check:
            compile_report()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
