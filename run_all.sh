#!/usr/bin/env bash
# Rebuild everything from the real corpus: fetch the PX4 User Guide (pinned commit), index, evaluate, figures.
set -e
python -m docrag.corpus          # sparse clone of PX4/PX4-user_guide at commit 2fa62338 (about 90 MB)
python -m docrag.ingest_md       # chunks, images, OCR (cached in data/index/ocr_cache.json)
python -m docrag.evaluate        # 5-fold cross-validation by page on the 70 handwritten questions (about 3 min)
python -m docrag.plots
