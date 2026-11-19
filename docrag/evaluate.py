"""Evaluate on the PX4 User Guide with the handwritten questions.

Protocol: 5-fold cross-validation grouped by page (all the questions about one page are in the same fold), so the
reranker and the figure model are always tested on pages they never saw during training. Every question is tested
exactly once; 95% intervals are Wilson intervals over the questions.
"""
from __future__ import annotations

import json
import math
import pickle
import random
import statistics
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .pipeline import Pipeline, best_sentence, relevant
from .retrieval import features, tokens

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/index"


def load_questions(P: Pipeline) -> list[dict]:
    qs = yaml.safe_load((ROOT / "data/questions_px4.yaml").read_text())
    page_of = {c.id: c.section for c in P.r.chunks}
    for q in qs:
        q["question"] = q.pop("q")
        q["figures"] = sorted(f.id for f in P.r.figures if q.get("fig") and q["fig"].lower() in f.caption.lower()
                              and q["page"] in {page_of[c] for c in f.cited_by}) if q.get("fig") else []
    return qs


def check_questions(P: Pipeline, qs):
    """Every question must have at least one relevant passage, and every requested figure must exist."""
    bad = []
    for q in qs:
        if not any(relevant(c, q) for c in P.r.chunks):
            bad.append((q["question"], "no passage"))
        if q.get("fig") and not q["figures"]:
            bad.append((q["question"], "no figure"))
    return bad


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def train(P: Pipeline, qs):
    r = P.r
    r.reranker = r.fig_model = None
    X, y = [], []
    for q in qs:
        for rank, h in enumerate(r.candidates(r.expand(q["question"]))):
            X.append(features(q["question"], h, rank)); y.append(int(relevant(h.chunk, q)))
    r.reranker = model().fit(np.array(X), np.array(y))
    Xf, yf, rows = [], [], []
    for q in qs:
        hits = r.search(q["question"], k=5)
        sc = r.figure_scores(q["question"], hits)
        rows.append((q, sc))
        for f, x in sc:
            Xf.append(x); yf.append(int(f.id in q["figures"]))
    r.fig_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(np.array(Xf),
                                                                                            np.array(yf))
    best = (-1, 0.5)                       # display threshold: best F1 of the images shown, on the training pages
    probs = [(q, [(f.id, p) for (f, _), p in zip(sc, r.fig_model.predict_proba(np.stack([x for _, x in sc]))[:, 1])])
             for q, sc in rows]
    for t in np.concatenate([np.arange(0.05, 0.95, 0.05), [0.96, 0.98, 0.99, 0.995]]):
        shown = good = need = found = 0
        for q, fp in probs:
            ids = [i for i, p in sorted(fp, key=lambda z: -z[1])[:2] if p >= t]
            shown += len(ids); good += sum(i in q["figures"] for i in ids)
            if q["figures"]:
                need += 1; found += any(i in q["figures"] for i in ids)
        pr, rc = good / max(shown, 1), found / max(need, 1)
        best = max(best, (2 * pr * rc / max(pr + rc, 1e-9), float(t)))
    P.threshold = best[1]
    return P.threshold


def evaluate_question(P: Pipeline, q, modes):
    out = {}
    for m in modes:
        hits = P.r.search(q["question"], k=10, mode=m)
        ranks = [i for i, h in enumerate(hits) if relevant(h.chunk, q)]
        out[m] = ranks[0] if ranks else None
    hits = P.r.search(q["question"], k=5)
    sent, _ = best_sentence(q["question"], hits)
    out["answer_ok"] = q["key"].lower() in sent.lower()
    out["top_page"] = hits[0].chunk.section
    out["top_title"] = hits[0].chunk.title
    learned = [f.id for f, _ in P.r.figures_for(q["question"], hits, k=2, threshold=P.threshold)]
    top_passage = list(hits[0].chunk.figure_refs)[:2]
    s = P.r.fig_bm25.get_scores(tokens(q["question"]))
    direct = [P.r.figures[int(np.argmax(s))].id]
    out["images"] = dict(learned=learned, top_passage=top_passage, direct=direct)
    return out


def summarise(qs, res, modes):
    n = len(qs)
    retr = {}
    for m in modes:
        r1 = sum(res[i][m] == 0 for i in range(n)); r5 = sum(res[i][m] is not None and res[i][m] < 5 for i in range(n))
        mrr = sum(1 / (1 + res[i][m]) for i in range(n) if res[i][m] is not None) / n
        retr[m] = dict(recall_at_1=round(r1 / n, 3), recall_at_1_ci95=wilson(r1, n), recall_at_5=round(r5 / n, 3),
                       mrr_at_10=round(mrr, 3))
    imgs = {}
    need = [i for i, q in enumerate(qs) if q["figures"]]
    for meth in ("top_passage", "direct", "learned"):
        shown = sum(len(res[i]["images"][meth]) for i in range(n))
        good = sum(sum(x in qs[i]["figures"] for x in res[i]["images"][meth]) for i in range(n))
        found = sum(any(x in qs[i]["figures"] for x in res[i]["images"][meth]) for i in need)
        imgs[meth] = dict(precision=round(good / max(shown, 1), 3), recall=round(found / max(len(need), 1), 3),
                          images_per_answer=round(shown / n, 2), questions_with_a_figure=len(need))
    ans = sum(res[i]["answer_ok"] for i in range(n))
    return dict(n_questions=n, retrieval=retr, images=imgs,
                answer_sentence_contains_key=dict(rate=round(ans / n, 3), ci95=wilson(ans, n)))


def crossval(P: Pipeline, qs, folds=5, seed=0, modes=("bm25", "dense", "hybrid", "rerank")):
    pages = sorted({q["page"] for q in qs})
    random.Random(seed).shuffle(pages)
    fold_of = {p: i % folds for i, p in enumerate(pages)}
    res = [None] * len(qs)
    thresholds = []
    for k in range(folds):
        tr = [q for q in qs if fold_of[q["page"]] != k]
        thresholds.append(train(P, tr))
        for i, q in enumerate(qs):
            if fold_of[q["page"]] == k:
                res[i] = evaluate_question(P, q, modes)
    return res, thresholds


def latency(P: Pipeline, qs, reps=3):
    out = {}
    for with_img in (False, True):
        ts = []
        for _ in range(reps):
            for q in qs:
                t = time.perf_counter(); P.ask(q["question"], with_images=with_img); ts.append(time.perf_counter() - t)
        ts.sort()
        out["with_images" if with_img else "text_only"] = dict(p50_ms=round(1000 * statistics.median(ts), 1),
                                                               p95_ms=round(1000 * ts[int(0.95 * len(ts)) - 1], 1))
    return out


def main():
    P = Pipeline(INDEX, models=INDEX / "none.pkl")
    qs = load_questions(P)
    bad = check_questions(P, qs)
    assert not bad, bad
    corpus = dict(pages=len({c.section for c in P.r.chunks}), chunks=len(P.r.chunks), figures=len(P.r.figures),
                  figures_with_ocr_text=sum(1 for f in P.r.figures if f.ocr),
                  words=sum(len(c.text.split()) for c in P.r.chunks))
    res, thr = crossval(P, qs)
    out = dict(corpus=corpus, protocol="5-fold cross-validation grouped by page, every question tested once",
               **summarise(qs, res, ("bm25", "dense", "hybrid", "rerank")),
               thresholds=[round(t, 2) for t in thr])
    out["by_type"] = {}
    for t in sorted({q["type"] for q in qs}):
        idx = [i for i, q in enumerate(qs) if q["type"] == t]
        s = summarise([qs[i] for i in idx], [res[i] for i in idx], ("bm25", "rerank"))
        out["by_type"][t] = dict(n=len(idx), bm25_r1=s["retrieval"]["bm25"]["recall_at_1"],
                                 rerank_r1=s["retrieval"]["rerank"]["recall_at_1"],
                                 answer=s["answer_sentence_contains_key"]["rate"])
    # stability: same protocol with four other fold assignments
    runs = []
    for seed in (1, 2, 3, 4):
        r2, _ = crossval(P, qs, seed=seed, modes=("rerank",))
        s = summarise(qs, r2, ("rerank",))
        runs.append(dict(rerank_r1=s["retrieval"]["rerank"]["recall_at_1"],
                         img_precision=s["images"]["learned"]["precision"], img_recall=s["images"]["learned"]["recall"]))
    out["other_fold_seeds"] = runs
    # ablation: chunks without their "Page > Heading" context
    from .ingest_md import ingest
    nc = ROOT / "data/index_nocontext"
    import shutil
    nc.mkdir(exist_ok=True); shutil.copy(INDEX / "ocr_cache.json", nc / "ocr_cache.json")
    ingest(ROOT / "data/px4", nc, context=False)
    Pn = Pipeline(nc, models=nc / "none.pkl")
    qn = load_questions(Pn)
    rn, _ = crossval(Pn, qn, modes=("bm25", "rerank"))
    sn = summarise(qn, rn, ("bm25", "rerank"))
    out["ablation_no_context_headers"] = dict(bm25_r1=sn["retrieval"]["bm25"]["recall_at_1"],
                                              rerank_r1=sn["retrieval"]["rerank"]["recall_at_1"],
                                              answer=sn["answer_sentence_contains_key"]["rate"])
    # final models on all questions, for the app
    thr_all = train(P, qs)
    out["latency"] = latency(P, qs)
    (INDEX / "models.pkl").write_bytes(pickle.dumps(dict(reranker=P.r.reranker, fig_model=P.r.fig_model,
                                                         threshold=thr_all)))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/metrics.json").write_text(json.dumps(out, indent=1))
    per_q = [dict(question=q["question"], type=q["type"], page=q["page"], key=q["key"], **{
        k: v for k, v in r.items()}) for q, r in zip(qs, res)]
    (ROOT / "results/per_question.json").write_text(json.dumps(per_q, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
