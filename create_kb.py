# build_kb.py
"""
Build a chunked KB JSON from a raw zoning_plan_*.json.

Rules applied:
- Keep only document_type in {"Bestemmingsplan", "Omgevingsplan"}
- Ignore any doc with "Parapluplan" in the title
- Ignore temporary parts implicitly (we only use zoning_documents[].text)
- Chunk by Artikel headings (Markdown headings containing 'Artikel')
- Preserve heading_path hierarchy and add permit-free flag per chunk
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json, re, hashlib


HEADING_LINE_RE = re.compile(r"^\s*(#{1,6})\s+(.*\S)\s*$")
ARTIKEL_RE = re.compile(r"\bartikel\b", re.IGNORECASE)
PERMIT_FREE_RE = re.compile(
    r"\b(vergunningvrije|mag\s+zonder\s+omgevingsvergunning|toegestaan\s+zonder\s+omgevingsvergunning)\b",
    re.IGNORECASE
)

def stable_id(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:12]

def normalize_ws(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_big_chunk(text: str, max_chars: int = 9000) -> List[str]:
    """Safety valve if an Artikel is huge: split on blank lines."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, buf, buf_len = [], [], 0
    for p in parts:
        add_len = len(p) + (2 if buf else 0)
        if buf and buf_len + add_len > max_chars:
            out.append("\n\n".join(buf).strip())
            buf, buf_len = [], 0
        buf.append(p)
        buf_len += add_len
    if buf:
        out.append("\n\n".join(buf).strip())
    return out

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    established_date: str
    heading_path: List[str]
    text: str
    contains_permit_free_language: bool

def chunk_by_artikel(doc_id: str, doc_title: str, doc_type: str, established_date: str, raw_text: str) -> List[Chunk]:
    raw_text = normalize_ws(raw_text)
    if not raw_text:
        return []

    stack: List[Tuple[int, str]] = []
    heading_path: List[str] = []
    chunks: List[Chunk] = []

    inside_artikel = False
    buf: List[str] = []

    def set_stack(level: int, title: str) -> None:
        nonlocal stack, heading_path
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = [t for _, t in stack]

    def flush() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        if not text:
            buf = []
            return
        for piece_i, piece in enumerate(split_big_chunk(text)):
            cid = stable_id(doc_id, str(len(chunks)), str(piece_i), piece[:120])
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                doc_title=doc_title,
                doc_type=doc_type,
                established_date=established_date or "",
                heading_path=heading_path[:],
                text=piece,
                contains_permit_free_language=bool(PERMIT_FREE_RE.search(piece)),
            ))
        buf = []

    for line in raw_text.split("\n"):
        m = HEADING_LINE_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            set_stack(level, title)

            if ARTIKEL_RE.search(title):
                # new Artikel boundary
                if inside_artikel and buf:
                    flush()
                inside_artikel = True
                buf.append(line)
            else:
                if inside_artikel:
                    buf.append(line)
            continue

        if inside_artikel:
            buf.append(line)

    if buf:
        flush()

    # Fallback: if no Artikel headings found
    if not chunks:
        cid = stable_id(doc_id, "fallback", raw_text[:120])
        chunks.append(Chunk(
            chunk_id=cid,
            doc_id=doc_id,
            doc_title=doc_title,
            doc_type=doc_type,
            established_date=established_date or "",
            heading_path=[],
            text=raw_text,
            contains_permit_free_language=bool(PERMIT_FREE_RE.search(raw_text)),
        ))

    return chunks

def build_kb(raw: Dict[str, Any]) -> Dict[str, Any]:
    docs_in = raw.get("zoning_documents", [])
    docs = []
    for d in docs_in:
        if d.get("document_type") not in {"Bestemmingsplan", "Omgevingsplan"}:
            continue
        if "parapluplan" in (d.get("title", "").lower()):
            continue
        docs.append(d)

    meta = raw.get("zoning_metadata", {}) or {}
    out_docs = []
    for d in docs:
        chunks = chunk_by_artikel(
            doc_id=d.get("id", ""),
            doc_title=d.get("title", ""),
            doc_type=d.get("document_type", ""),
            established_date=d.get("established_date", ""),
            raw_text=d.get("text", "") or "",
        )
        out_docs.append({
            "id": d.get("id", ""),
            "title": d.get("title", ""),
            "document_type": d.get("document_type", ""),
            "established_date": d.get("established_date", ""),
            "chunks": [asdict(c) for c in chunks],
        })

    return {
        "address": raw.get("address", {}) or {},
        "bestemmingsvlakken": meta.get("bestemmingsvlakken", []) or [],
        "maatvoeringen": meta.get("maatvoeringen", []) or [],
        "documents": out_docs,
    }

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True, help="raw zoning_plan_*.json")
    p.add_argument("--out", required=True, help="output chunked KB json")
    args = p.parse_args()

    raw = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    kb = build_kb(raw)
    Path(args.out).write_text(json.dumps(kb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote chunked KB: {args.out}")

if __name__ == "__main__":
    main()
