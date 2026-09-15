"""Print text from the official DOCX reference files without external dependencies."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

REFERENCE_DIR = Path("docs/reference")
FILES = [
    "Descripcion_general_variables_DataSet.docx",
    "Diccionario_de_indices_satelitales.docx",
]
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_docx_paragraphs(path: Path) -> list[str]:
    """Extract visible paragraph/table text from a DOCX document.xml file."""

    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", NS):
        text = "".join(
            node.text or "" for node in paragraph.findall(".//w:t", NS)
        ).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def main() -> None:
    """Print each reference document with line numbers for source audit."""

    for filename in FILES:
        path = REFERENCE_DIR / filename
        print(f"=== {filename} ===")
        for index, text in enumerate(extract_docx_paragraphs(path), start=1):
            print(f"{index:04d}: {text}")
        print()


if __name__ == "__main__":
    main()
