"""Question answering pipeline: retrieve, rerank, pick the answer sentence, attach the right figures.

The answer is extractive by default: the sentence of the retrieved passages that best matches the question,
quoted with its section. If an OpenAI-compatible endpoint is configured (Azure OpenAI or any other, through
LLM_BASE_URL, LLM_API_KEY and LLM_MODEL), the passages are sent to the model with an instruction to answer
only from them and to cite the section; the extractive answer stays the fallback.
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import re
import time
from pathlib import Path

from .ingest import load
from .retrieval import INTENTS, Retriever, intent_vector, tokens

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def relevant(chunk, q: dict) -> bool:
    """A passage answers a question if it is on the page that holds the answer and contains the answer key."""
    return chunk.section == q["page"] and q["key"].lower() in (chunk.title + " " + chunk.text).lower()


def best_sentence(question: str, hits) -> tuple[str, str]:
    iv = intent_vector(question)
    qt = set(tokens(question))
    best, best_s, best_sec = -1.0, "", ""
    for rank, h in enumerate(hits[:3]):
        sents = SENT.split(h.chunk.text)
        for s in sents:
            sl = s
            score = len(qt & set(tokens(s))) / max(len(qt), 1) + 0.6 * h.score - 0.15 * rank
            for k, (_, pat) in INTENTS.items():
                if iv[k] and re.search(pat, sl):
                    score += 0.5
            if score > best:
                best, best_s, best_sec = score, s.strip(), f"{h.chunk.title} ({h.chunk.section})"
    return best_s, best_sec


def llm_answer(question: str, hits) -> str | None:
    base, key, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY"), os.getenv("LLM_MODEL")
    if not (base and key and model):
        return None
    import urllib.request
    ctx = "\n\n".join(f"[{h.chunk.section} {h.chunk.title}] {h.chunk.text}" for h in hits[:4])
    body = {"model": model, "temperature": 0, "messages": [
        {"role": "system", "content": "You answer technician questions using only the passages given. Cite the "
                                      "page in brackets. If the passages do not contain the answer, say so."},
        {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}"}]}
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001  (the extractive answer is the fallback)
        return None


class Pipeline:
    def __init__(self, index_dir: Path, models: Path | None = None):
        self.index_dir = Path(index_dir)
        chunks, figures = load(self.index_dir)
        self.r = Retriever(chunks, figures)
        models = models or self.index_dir / "models.pkl"
        if Path(models).exists():
            m = pickle.loads(Path(models).read_bytes())
            self.r.reranker, self.r.fig_model = m["reranker"], m["fig_model"]
            self.threshold = m.get("threshold", 0.5)
        else:
            self.threshold = 0.5

    def ask(self, question: str, with_images: bool = True, k: int = 5) -> dict:
        t0 = time.perf_counter()
        hits = self.r.search(question, k=k)
        sent, sec = best_sentence(question, hits)
        t1 = time.perf_counter()
        images = []
        if with_images:
            for f, p in self.r.figures_for(question, hits, threshold=self.threshold):
                data = (self.index_dir / "figures" / f.path).read_bytes()
                mime = "image/jpeg" if f.path.lower().endswith((".jpg", ".jpeg")) else (
                    "image/gif" if f.path.lower().endswith(".gif") else "image/png")
                images.append(dict(id=f.id, mime=mime, caption=f.caption, page=f.page, section=f.section, score=round(p, 3),
                                   png_base64=base64.b64encode(data).decode()))
        t2 = time.perf_counter()
        gen = llm_answer(question, hits)
        return dict(question=question, answer=gen or sent, answer_source=sec,
                    mode="llm" if gen else "extractive",
                    passages=[dict(id=h.chunk.id, section=h.chunk.section, title=h.chunk.title, page=h.chunk.page,
                                   score=round(h.score, 3), text=h.chunk.text, figure_refs=h.chunk.figure_refs)
                              for h in hits],
                    images=images, latency_ms=dict(text=round((t1 - t0) * 1000, 1),
                                                   images=round((t2 - t1) * 1000, 1)))
