"""Parse the PX4 User Guide (Markdown) into text chunks and figures.

Chunks follow the structure of the guide: one heading = one chunk (split at about 120 words, on paragraph
boundaries), and every chunk carries its context, "Page title > Heading", so a passage about "Wiring" still says
which flight controller it is about. Figures are the images the pages embed (Markdown ![alt](src) or HTML
<img src title>): their caption is the alt text or title, the text inside them is read by OCR (Tesseract), and they
are linked to the chunk that embeds them.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from .ingest import Chunk, Figure
from .corpus import EXCLUDE

HEAD = re.compile(r"^(#{1,4})\s+(.*?)\s*(\{#[^}]*\})?\s*$")
MD_IMG = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)")
HTML_IMG = re.compile(r"<img\s[^>]*?src=\"([^\"]+)\"[^>]*>")
TITLE_ATTR = re.compile(r"(?:title|alt)=\"([^\"]*)\"")
LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
TAGS = re.compile(r"<[^>]+>")
RASTER = (".png", ".jpg", ".jpeg", ".gif")


def ocr(path: Path, cache: dict) -> str:
    key = hashlib.md5(path.read_bytes()).hexdigest()
    if key not in cache:
        try:
            r = subprocess.run(["tesseract", str(path), "-", "--psm", "11"], capture_output=True, text=True,
                               timeout=60)
            cache[key] = " ".join(re.findall(r"[A-Za-z][A-Za-z0-9\-/]{2,}", r.stdout))
        except (OSError, subprocess.TimeoutExpired):
            cache[key] = ""
    return cache[key]


def clean(text: str) -> str:
    text = MD_IMG.sub("", text)
    text = HTML_IMG.sub("", text)
    text = LINK.sub(r"\1", text)
    text = TAGS.sub(" ", text)
    text = re.sub(r"^:::\s*\w*", "", text, flags=re.M)         # VitePress callouts (::: tip / ::: warning)
    text = text.replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def images_in(md_text: str, md_path: Path, root: Path):
    out = []
    for m in MD_IMG.finditer(md_text):
        out.append((m.group(1) or m.group(3) or "", m.group(2)))
    for m in HTML_IMG.finditer(md_text):
        t = TITLE_ATTR.search(m.group(0))
        out.append((t.group(1) if t else "", m.group(1)))
    res = []
    for alt, src in out:
        p = (md_path.parent / src.split("#")[0]).resolve()
        if p.suffix.lower() in RASTER and p.exists() and str(p).startswith(str(root.resolve())):
            res.append((alt.strip(), p))
    return res


def ingest(root: Path, out: Path, max_words: int = 120, context: bool = True, run_ocr: bool = True):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cache_f = out / "ocr_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    chunks: list[Chunk] = []
    figures: dict[str, Figure] = {}
    for md in sorted((root / "en").rglob("*.md")):
        page = str(md.relative_to(root / "en"))
        if page in EXCLUDE or md.name in ("SUMMARY.md", "_sidebar.md", "index.md"):
            continue
        text = md.read_text(errors="ignore")
        text = re.sub(r"```.*?```", lambda m: m.group(0).replace("\n", " ⏎ "), text, flags=re.S)
        title, head, paras, cur = page, "", [], []

        def flush_section():
            nonlocal paras
            buf: list[str] = []
            imgs: list = []
            pieces = []
            for p in paras:                   # long tables and lists are split on line boundaries
                if len(clean(p).split()) <= max_words:
                    pieces.append(p); continue
                grp: list[str] = []
                for line in p.splitlines():
                    if grp and len(clean(" ".join(grp + [line])).split()) > max_words:
                        pieces.append("\n".join(grp)); grp = []
                    grp.append(line)
                if grp:
                    pieces.append("\n".join(grp))
            for p in pieces + [None]:
                words = sum(len(x.split()) for x in buf)
                if p is None or (buf and words + len(clean(p).split()) > max_words):
                    body = clean(" ".join(buf)).replace(" ⏎ ", " ")
                    if body or imgs:
                        cid = f"c{len(chunks):05d}"
                        ctitle = f"{title} > {head}" if context and head and head != title else (head or title)
                        refs = []
                        for alt, path in imgs:
                            fid = str(path.relative_to(root.resolve()))
                            if fid not in figures:
                                flat = fid.replace("/", "__")
                                shutil.copy(path, out / "figures" / flat)
                                figures[fid] = Figure(fid, alt, page, head, title, flat,
                                                      ocr(path, cache) if run_ocr else "")
                            figures[fid].cited_by.append(cid)
                            refs.append(fid)
                        chunks.append(Chunk(cid, page, ctitle, len(chunks), body, sorted(set(refs))))
                    buf, imgs = [], []
                if p is not None:
                    buf.append(p)
                    imgs += images_in(p, md, root)
            paras = []

        for block in re.split(r"\n\s*\n", text):
            lines = block.strip().splitlines()
            if not lines:
                continue
            m = HEAD.match(lines[0])
            if m:
                flush_section()
                h = clean(m.group(2))
                if len(m.group(1)) == 1:
                    title = h
                head = h
                lines = lines[1:]
            if lines:
                paras.append("\n".join(lines))
        flush_section()
    cache_f.write_text(json.dumps(cache))
    figs = list(figures.values())
    (out / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=0, ensure_ascii=False))
    (out / "figures.json").write_text(json.dumps([asdict(f) for f in figs], indent=0, ensure_ascii=False))
    return chunks, figs


if __name__ == "__main__":
    ch, fi = ingest(Path("data/px4"), Path("data/index"))
    print(len(ch), "chunks,", len(fi), "figures,", sum(1 for f in fi if f.ocr), "with OCR text")
