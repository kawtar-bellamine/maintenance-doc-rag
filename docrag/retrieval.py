"""Hybrid retrieval over the documentation: BM25 + dense embeddings, fused, then reranked; figures scored from
the passages that cite them and from a direct search on their caption and on the text read inside them."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk, Figure

STOP = set("""a an the of to in on at for by with and or is are be it its this that these those from as into
what which how when where who whom why do does did should must can i we you my our your me show give tell
please need there their them then than any all each per after before about""".split())
TOKEN = re.compile(r"[a-z0-9]+(?:[.,/_-][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    out = []
    for t in TOKEN.findall(text.lower()):
        if t in STOP:
            continue
        out.append(t)
        if "_" in t:                      # EKF2_GPS_POS_X is also found by "gps pos"
            out += [p for p in t.split("_") if p and p not in STOP]
    return out


# ---------- dense embeddings: SIF-weighted mean of spaCy word vectors ----------
class Embedder:
    """Smooth inverse frequency sentence embeddings (Arora et al., 2017) on en_core_web_md vectors,
    with the first principal component removed. Runs on CPU in a few milliseconds per query."""

    def __init__(self, corpus: list[str], a: float = 1e-3):
        import spacy
        self.nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner", "lemmatizer",
                                                          "attribute_ruler", "senter"])
        counts = Counter(t for doc in corpus for t in tokens(doc))
        total = sum(counts.values())
        self.w = {t: a / (a + c / total) for t, c in counts.items()}
        self.default_w = 1.0
        self.a = a
        raw = np.stack([self._raw(d) for d in corpus])
        _, _, vt = np.linalg.svd(raw, full_matrices=False)
        self.pc = vt[0]
        self.doc = self._post(raw)

    def _raw(self, text: str) -> np.ndarray:
        vecs, ws = [], []
        for t in tokens(text):
            lex = self.nlp.vocab[t]
            if lex.has_vector:
                vecs.append(lex.vector)
                ws.append(self.w.get(t, self.default_w))
        if not vecs:
            return np.zeros(self.nlp.vocab.vectors.shape[1], dtype=np.float32)
        return np.average(np.stack(vecs), axis=0, weights=ws)

    def _post(self, m: np.ndarray) -> np.ndarray:
        m = m - np.outer(m @ self.pc, self.pc)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(n, 1e-9)

    def encode(self, text: str) -> np.ndarray:
        return self._post(self._raw(text)[None])[0]


# ---------- query intent (used as reranker features) ----------
INTENTS = {   # (pattern in the question, pattern in the passage)
    "parameter": (r"parameter|which setting|set to", r"\b[A-Z][A-Z0-9]{1,}_[A-Z0-9_]{2,}"),
    "build": (r"\bbuild|\bflash|make target|firmware|bootloader", r"\bmake [a-z0-9_-]+"),
    "port": (r"where (do|does|should) .*(plug|connect|go)|which port|connect", r"\b(port|connector|TELEM|GPS\d|UART)"),
    "spec": (r"how (many|much)|voltage|weigh|processor|imu|which (gps|sensor|imu)|range|size", r"\d"),
    "fault": (r"why|refuse|error|fail|divergen|problem|not work|swapped|missing", r"caus|because|issue|check|warning"),
    "figure": (r"show|diagram|picture|view|pinout|drawing|overview", r"@@img"),
    "procedure": (r"how (do|should|long)|steps|calibrat", r"(^|\s)1\.|step|then"),
}


def intent_vector(q: str) -> dict[str, int]:
    ql = q.lower()
    return {k: int(bool(re.search(v[0], ql))) for k, v in INTENTS.items()}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float = 0.0
    dense: float = 0.0
    rrf: float = 0.0
    page: float = 0.0          # how well the page title matches the question (0..1)
    page_rank: int = 99        # rank of the page among the pages whose title matches best
    title_cov: float = 0.0     # share of the page-title words that the question contains


class Retriever:
    def __init__(self, chunks: list[Chunk], figures: list[Figure]):
        self.chunks, self.figures = chunks, figures
        self.docs = [f"{c.title}. {c.text}" for c in chunks]
        self.bm25 = BM25Okapi([tokens(d) for d in self.docs])
        self.emb = Embedder(self.docs)
        fig_docs = [f"{f.id} {f.caption} {f.title} {f.ocr}" for f in figures]
        self.fig_bm25 = BM25Okapi([tokens(d) for d in fig_docs])
        self.fig_by_id = {f.id: f for f in figures}
        # page routing: product documentation is organised by product, so the page title is a strong signal
        self.page_title = {}
        for c in chunks:
            self.page_title.setdefault(c.section, c.title.split(" > ")[0])
        self.page_list = list(self.page_title)
        self.page_bm25 = BM25Okapi([tokens(self.page_title[p] + " " + p.split("/")[-1].replace("_", " ")
                                           .replace(".md", "")) for p in self.page_list])
        self.chunks_of = {}
        for i, c in enumerate(chunks):
            self.chunks_of.setdefault(c.section, []).append(i)
        self.reranker = None
        self.fig_model = None
        self.glossary: dict[str, str] = {}

    def expand(self, q: str) -> str:
        """Append the manual's wording for any technician word found in the team glossary."""
        ql = q.lower()
        extra = [v for k, v in self.glossary.items() if re.search(r"\b" + re.escape(k) + r"\b", ql)]
        return q + (" (" + "; ".join(extra) + ")" if extra else "")

    # first stage
    def candidates(self, q: str, k: int = 30) -> list[Hit]:
        qt = tokens(q)
        b = self.bm25.get_scores(qt)
        d = self.emb.doc @ self.emb.encode(q)
        rb = {i: r for r, i in enumerate(np.argsort(-b))}
        rd = {i: r for r, i in enumerate(np.argsort(-d))}
        ps = self.page_bm25.get_scores(qt)
        pmax = max(ps.max(), 1e-9)
        prank = {self.page_list[j]: r for r, j in enumerate(np.argsort(-ps)[:5])}
        pscore = {p: ps[j] / pmax for j, p in enumerate(self.page_list)}
        pool = set(np.argsort(-b)[:k - 10]) | set(np.argsort(-d)[:5])
        for p in list(prank)[:3]:                 # the best chunks of the three best-titled pages
            idx = self.chunks_of[p]
            pool |= set(sorted(idx, key=lambda i: -b[i])[:4])
        rrf = {i: 1 / (60 + rb[i]) + 1 / (60 + rd[i]) for i in pool}
        top = sorted(pool, key=rrf.get, reverse=True)
        bmax = max(b.max(), 1e-9)
        qs = set(qt)
        out = []
        for i in top:
            c = self.chunks[i]
            tt = set(tokens(self.page_title[c.section]))
            out.append(Hit(c, rrf[i], b[i] / bmax, float(d[i]), rrf[i], pscore[c.section],
                           prank.get(c.section, 99), len(tt & qs) / max(len(tt), 1)))
        return out

    def search(self, q: str, k: int = 5, mode: str = "rerank") -> list[Hit]:
        q = self.expand(q)
        if mode == "bm25":
            s = self.bm25.get_scores(tokens(q))
            idx = np.argsort(-s)[:k]
            return [Hit(self.chunks[i], float(s[i])) for i in idx]
        if mode == "dense":
            s = self.emb.doc @ self.emb.encode(q)
            idx = np.argsort(-s)[:k]
            return [Hit(self.chunks[i], float(s[i])) for i in idx]
        hits = self.candidates(q)
        if mode == "hybrid" or self.reranker is None:
            return hits[:k]
        X = np.array([features(q, h, r) for r, h in enumerate(hits)])
        p = self.reranker.predict_proba(X)[:, 1]
        for h, s in zip(hits, p):
            h.score = float(s)
        return sorted(hits, key=lambda h: -h.score)[:k]

    # figures
    def figure_scores(self, q: str, hits: list[Hit]) -> list[tuple[Figure, np.ndarray]]:
        direct = self.fig_bm25.get_scores(tokens(q))
        dmax = max(direct.max(), 1e-9)
        out = []
        for j, f in enumerate(self.figures):
            ref_top1 = int(f.id in hits[0].chunk.figure_refs) if hits else 0
            ref_sum = sum(h.score / (1 + r) for r, h in enumerate(hits) if f.id in h.chunk.figure_refs)
            same_sec = max((h.score for h in hits[:3] if h.chunk.section == f.page), default=0.0)
            iv = intent_vector(q)
            x = np.array([direct[j] / dmax, ref_top1, ref_sum, same_sec, iv["figure"],
                          iv["figure"] * direct[j] / dmax, iv["figure"] * ref_top1])
            out.append((f, x))
        return out

    def figures_for(self, q: str, hits: list[Hit], k: int = 2, threshold: float = 0.5):
        q = self.expand(q)
        scored = self.figure_scores(q, hits)
        if self.fig_model is None:
            ranked = sorted(scored, key=lambda t: -(t[1][2] + t[1][0]))
            return [(f, float(x[2] + x[0])) for f, x in ranked[:k]]
        X = np.stack([x for _, x in scored])
        p = self.fig_model.predict_proba(X)[:, 1]
        order = np.argsort(-p)[:k]
        return [(scored[i][0], float(p[i])) for i in order if p[i] >= threshold]


def features(q: str, h: Hit, rank: int) -> list[float]:
    iv = intent_vector(q)
    raw = h.chunk.title + " " + h.chunk.text
    text = raw.lower()
    qt, tt = set(tokens(q)), set(tokens(h.chunk.title))
    f = [h.bm25, h.dense, h.rrf * 60, 1 / (1 + rank), len(qt & tt) / max(len(qt), 1),
         len(qt & set(tokens(text))) / max(len(qt), 1), math.log1p(len(text.split()))]
    for k, (_, doc_pat) in INTENTS.items():
        m = int(bool(h.chunk.figure_refs)) if doc_pat == r"@@img" else int(bool(re.search(doc_pat, raw)))
        f += [iv[k] * m]
    f += [int(bool(h.chunk.figure_refs)), h.page, 1 / (1 + h.page_rank), h.title_cov, h.title_cov * h.page]
    return f
