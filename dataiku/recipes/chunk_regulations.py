# Zone 7 - Regulations: splits the regulatory PDFs into passages for the Knowledge Bank.
# One row per passage, with its source document and page so that answers can cite them.
# Code env: creditops (pypdf).
import io
import re

import dataiku
import pandas as pd
from pypdf import PdfReader

CHUNK_SIZE = 1500     # characters, about 350 tokens
CHUNK_OVERLAP = 200
TITLES = {
    "EBA_GL_2020_06_octroi_suivi_prets_FR": ("Orientations EBA sur l'octroi et le suivi des prêts (EBA/GL/2020/06)", "fr"),
    "EBA_GL_2020_06_loan_origination_EN": ("EBA Guidelines on loan origination and monitoring (EBA/GL/2020/06)", "en"),
    "AI_Act_Reglement_2024_1689_FR": ("Règlement (UE) 2024/1689 sur l'intelligence artificielle (AI Act)", "fr"),
    "Directive_credit_consommation_2023_2225_FR": ("Directive (UE) 2023/2225 relative aux contrats de crédit aux consommateurs", "fr"),
    "ACPR_AS183_financement_habitat_2025": ("ACPR, Le financement de l'habitat en 2025 (AS n°183)", "fr"),
}


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)          # words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def split(text: str) -> list[str]:
    """Fixed-size passages that end on a sentence or line break when possible."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + CHUNK_SIZE // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 80]


folder = dataiku.Folder("reg_documents")
rows = []
for path in folder.list_paths_in_partition():
    if not path.lower().endswith(".pdf"):
        continue
    source = path.strip("/")[:-4]
    title, language = TITLES.get(source, (source, "fr"))
    with folder.get_download_stream(path) as stream:
        reader = PdfReader(io.BytesIO(stream.read()))
    for page_number, page in enumerate(reader.pages, start=1):
        for i, chunk in enumerate(split(clean(page.extract_text() or ""))):
            rows.append({"chunk_id": f"{source}_p{page_number}_{i}", "source": source,
                         "title": title, "language": language, "page": page_number, "text": chunk})

chunks = pd.DataFrame(rows)
print(chunks.groupby("source").size().to_string())
dataiku.Dataset("reg_chunks").write_with_schema(chunks)
