
import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
from pypdf import PdfReader

import vertexai
from vertexai.language_models import TextEmbeddingModel

EMBEDDING_MODEL_NAME = "text-embedding-004"


@dataclass
class Chunk:
    id: str
    doc_name: str
    source_type: str            # "pdf" | "text"
    page: Optional[int]         # PDF page number (1-indexed); None for text
    section: Optional[str]      # heading/section label when available
    chunk_index: int
    text: str
    embedding: List[float]


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, target_chars: int = 1100, overlap_chars: int = 200) -> List[str]:
    """
    Simple, predictable chunking for MVP.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + target_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return chunks


def read_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    out: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = clean_text(txt)
        if txt:
            out.append((i + 1, txt))  # 1-indexed pages for citations
    return out


def split_text_sections(text: str) -> List[Tuple[Optional[str], str]]:
    """
    Very simple 'section' detection for .txt/.md:
    - Lines ending in ":" or standalone Title Case lines become section labels.
    Falls back to a single section if nothing detected.
    """
    text = clean_text(text)
    if not text:
        return []

    lines = [ln.strip() for ln in text.split("\n")]
    sections: List[Tuple[Optional[str], List[str]]] = []
    current_title: Optional[str] = None
    current_body: List[str] = []

    def flush():
        nonlocal current_title, current_body
        body = clean_text("\n".join(current_body))
        if body:
            sections.append((current_title, body))
        current_title = None
        current_body = []

    for ln in lines:
        if not ln:
            current_body.append("")
            continue

        is_heading = False
        if ln.endswith(":") and len(ln) <= 80:
            is_heading = True
        elif len(ln) <= 60 and ln == ln.title() and ln.isalpha() is False:
            # loose heuristic; keep minimal to avoid weird splits
            pass

        if is_heading:
            flush()
            current_title = ln.rstrip(":").strip()
        else:
            current_body.append(ln)

    flush()

    if not sections:
        return [(None, text)]
    return [(title, body) for title, body in sections]


def embed_texts(model: TextEmbeddingModel, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = model.get_embeddings(batch)
        embeddings.extend([r.values for r in res])
    return embeddings


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_chunks_from_docs(docs_dir: Path, target_chars: int, overlap_chars: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    metas: List[Dict[str, Any]] = []
    texts: List[str] = []

    docs = sorted([p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() in [".pdf", ".txt", ".md"]])
    if not docs:
        raise SystemExit(f"No docs found in {docs_dir}. Add PDF/TXT/MD files to proceed.")

    for path in docs:
        doc_name = path.name
        ext = path.suffix.lower()

        if ext == ".pdf":
            pages = read_pdf_pages(path)
            for page_num, page_text in pages:
                chunks = chunk_text(page_text, target_chars, overlap_chars)
                for ci, ch in enumerate(chunks):
                    chunk_id = f"{doc_name}::p{page_num}::c{ci}"
                    metas.append({
                        "id": chunk_id,
                        "doc_name": doc_name,
                        "source_type": "pdf",
                        "page": page_num,
                        "section": None,
                        "chunk_index": ci,
                        "text": ch,
                    })
                    texts.append(ch)

        else:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            sections = split_text_sections(raw)
            for si, (title, body) in enumerate(sections):
                chunks = chunk_text(body, target_chars, overlap_chars)
                for ci, ch in enumerate(chunks):
                    sec = title or None
                    chunk_id = f"{doc_name}::s{si}::{('sec-'+sec) if sec else 'nosec'}::c{ci}"
                    metas.append({
                        "id": chunk_id,
                        "doc_name": doc_name,
                        "source_type": "text",
                        "page": None,
                        "section": sec,
                        "chunk_index": ci,
                        "text": ch if not sec else f"[{sec}]\n{ch}",
                    })
                    texts.append(metas[-1]["text"])

    if not texts:
        raise SystemExit("No extractable text found. If your PDFs are scanned images, we’ll need a different extractor.")
    return metas, texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", required=True)
    parser.add_argument("--out", default="index.json")
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="europe-west2")
    parser.add_argument("--target_chars", type=int, default=1100)
    parser.add_argument("--overlap_chars", type=int, default=200)
    parser.add_argument("--sample_query", default="")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise SystemExit(f"Docs dir not found: {docs_dir}")

    # Vertex init uses Application Default Credentials (ADC)
    vertexai.init(project=args.project, location=args.location)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

    print(f"Reading + chunking docs in: {docs_dir}")
    metas, texts = build_chunks_from_docs(docs_dir, args.target_chars, args.overlap_chars)
    print(f"Created {len(texts)} chunks. Embedding with {EMBEDDING_MODEL_NAME}...")

    embeddings = embed_texts(model, texts, batch_size=32)

    chunks: List[Chunk] = []
    for meta, emb in zip(metas, embeddings):
        chunks.append(Chunk(
            id=meta["id"],
            doc_name=meta["doc_name"],
            source_type=meta["source_type"],
            page=meta["page"],
            section=meta["section"],
            chunk_index=meta["chunk_index"],
            text=meta["text"],
            embedding=emb,
        ))

    index_obj = {
        "schema_version": 1,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "project": args.project,
        "location": args.location,
        "chunking": {"target_chars": args.target_chars, "overlap_chars": args.overlap_chars},
        "chunks": [asdict(c) for c in chunks],
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(index_obj, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    if args.sample_query.strip():
        print("\n--- Retrieval test ---")
        q = args.sample_query.strip()
        q_emb = model.get_embeddings([q])[0].values
        qv = np.array(q_emb, dtype=np.float32)

        scored: List[Tuple[float, Chunk]] = []
        for c in chunks:
            cv = np.array(c.embedding, dtype=np.float32)
            scored.append((cosine_sim(qv, cv), c))
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, c in scored[:args.top_k]:
            cite = f"{c.doc_name}"
            if c.page is not None:
                cite += f" p.{c.page}"
            if c.section:
                cite += f" [{c.section}]"
            print(f"\nscore={score:.4f}  {cite}")
            print(c.text[:450] + ("..." if len(c.text) > 450 else ""))


if __name__ == "__main__":
    main()
