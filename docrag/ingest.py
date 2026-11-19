"""Data model (Chunk, Figure) and the PDF path: parse a manual PDF into text chunks and figures (pip install pymupdf).
The PX4 evaluation uses the Markdown path, ingest_md.py.

Nothing here uses the generator's data model: the pipeline only sees the PDF, as it would with a real
manual. For every figure it keeps the image, its caption, its page and section, the text read inside the
image by OCR, and the passages that cite it.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path


HEAD_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\s+(\S.*)$")
FIG_RE = re.compile(r"Figure\s+(\d+-\d+)")


@dataclass
class Chunk:
    id: str
    section: str
    title: str
    page: int
    text: str
    figure_refs: list[str] = field(default_factory=list)


@dataclass
class Figure:
    id: str            # "Figure 4-3"
    caption: str
    page: int
    section: str
    title: str
    path: str
    ocr: str
    cited_by: list[str] = field(default_factory=list)


def _blocks(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        spans = [s for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        text = " ".join(" ".join(s["text"] for s in line["spans"]).strip() for line in b["lines"]).strip()
        size = max(s["size"] for s in spans)
        bold = any("Bold" in s["font"] for s in spans)
        out.append(dict(text=re.sub(r"\s+", " ", text), size=size, bold=bold, bbox=b["bbox"]))
    return out


def ocr(path: Path) -> str:
    try:
        r = subprocess.run(["tesseract", str(path), "-", "--psm", "11"], capture_output=True, text=True,
                           timeout=60)
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-/]{2,}", r.stdout)
        return " ".join(words)
    except (OSError, subprocess.TimeoutExpired):
        return ""


def ingest(pdf: Path, out: Path, max_words: int = 110, context: bool = True) -> tuple[list[Chunk], list[Figure]]:
    import pymupdf
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    doc = pymupdf.open(pdf)
    section, title = "0", "Front matter"
    paras: list[dict] = []                     # paragraphs with their section
    figures: list[Figure] = []
    titles: dict[str, str] = {}
    for pno, page in enumerate(doc, start=1):
        blocks = _blocks(page)
        imgs = []
        for img in page.get_images(full=True):
            for rect in page.get_image_rects(img[0]):
                imgs.append((img[0], rect))
        caption_blocks = set()
        for xref, rect in imgs:
            below = [b for b in blocks if b["bbox"][1] >= rect.y1 - 2 and b["bbox"][1] - rect.y1 < 30
                     and FIG_RE.match(b["text"])]
            if not below:
                continue
            cap = below[0]
            caption_blocks.add(id(cap))
            fid = "Figure " + FIG_RE.match(cap["text"]).group(1)
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            path = out / "figures" / (fid.replace(" ", "_") + ".png")
            pix.save(path)
            figures.append(Figure(fid, cap["text"].split(". ", 1)[-1], pno, "", "", str(path.name), ocr(path)))
            figures[-1]._y = rect.y0  # type: ignore[attr-defined]
        # walk text in reading order, tracking sections
        events = [("b", b["bbox"][1], b) for b in blocks if id(b) not in caption_blocks]
        events += [("f", f._y, f) for f in figures if f.page == pno]  # type: ignore[attr-defined]
        for kind, _, obj in sorted(events, key=lambda e: e[1]):
            if kind == "f":
                obj.section, obj.title = section, title
                continue
            t = obj["text"]
            m = HEAD_RE.match(t)
            if m and obj["bold"] and obj["size"] >= 10 and len(t) < 90:
                section, title = m.group(1), m.group(2)
                titles[section] = title
                parent = section.rsplit(".", 1)[0]
                if context and "." in section and parent in titles and section.count(".") >= 2:
                    # contextual header: a sub-section chunk keeps the name of what it is about
                    title = f"{titles[parent]} > {title}"
                continue
            if t.startswith("HPU-200 CMM") or re.fullmatch(r"Page \d+", t):
                continue
            paras.append(dict(section=section, title=title, page=pno, text=t))
    # chunks: consecutive paragraphs of the same section, up to max_words
    chunks: list[Chunk] = []
    cur: list[dict] = []

    def flush():
        if not cur:
            return
        text = " ".join(p["text"] for p in cur)
        cid = f"c{len(chunks):04d}"
        chunks.append(Chunk(cid, cur[0]["section"], cur[0]["title"], cur[0]["page"], text,
                            sorted(set("Figure " + x for x in FIG_RE.findall(text)))))
        cur.clear()

    for p in paras:
        n_cur = sum(len(x["text"].split()) for x in cur)
        own = p["text"].startswith("Fault ")      # structure-aware: one fault entry per chunk
        if cur and (own or p["section"] != cur[0]["section"] or n_cur + len(p["text"].split()) > max_words):
            flush()
        cur.append(p)
    flush()
    by_id = {f.id: f for f in figures}
    for c in chunks:
        for r in c.figure_refs:
            if r in by_id:
                by_id[r].cited_by.append(c.id)
    for f in figures:
        if hasattr(f, "_y"):
            del f._y
    (out / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1, ensure_ascii=False))
    (out / "figures.json").write_text(json.dumps([asdict(f) for f in figures], indent=1, ensure_ascii=False))
    return chunks, figures


def load(out: Path):
    chunks = [Chunk(**c) for c in json.loads((out / "chunks.json").read_text())]
    figures = [Figure(**f) for f in json.loads((out / "figures.json").read_text())]
    return chunks, figures


if __name__ == "__main__":
    import sys
    ch, fi = ingest(Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "data/index_pdf"))
    print(len(ch), "chunks,", len(fi), "figures,", sum(1 for f in fi if f.ocr), "with OCR text")
