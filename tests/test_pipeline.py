"""Tests on the real PX4 User Guide index (run python -m docrag.corpus, docrag.ingest_md, docrag.evaluate first)."""
from pathlib import Path

import pytest
import yaml

from docrag.ingest import load
from docrag.ingest_md import clean
from docrag.pipeline import Pipeline, relevant
from docrag.retrieval import intent_vector, tokens

ROOT = Path(__file__).resolve().parents[1]
IDX = ROOT / "data/index"


@pytest.fixture(scope="module")
def pipe():
    return Pipeline(IDX)


def test_index_covers_the_guide_without_the_parameter_reference():
    chunks, figures = load(IDX)
    pages = {c.section for c in chunks}
    assert len(pages) > 120 and len(figures) > 400
    assert "advanced_config/parameter_reference.md" not in pages
    for f in figures:
        assert (IDX / "figures" / f.path).exists() and f.cited_by


def test_chunks_carry_page_and_heading():
    chunks, _ = load(IDX)
    wiring = [c for c in chunks if c.section == "assembly/quick_start_pixhawk4.md" and "Telemetry" in c.title]
    assert wiring and all(c.title.startswith("Pixhawk 4 Wiring Quick Start > ") for c in wiring)
    assert max(len(c.text.split()) for c in chunks) < 250


def test_markdown_cleaning():
    assert clean("See [Pixhawk 4](../flight_controller/pixhawk4.md) ![x](a.png) `EKF2_GPS_POS_X`") == \
        "See Pixhawk 4 EKF2_GPS_POS_X"


def test_tokens_keep_parameters_and_devices():
    t = tokens("Set EKF2_GPS_POS_X, TELEM1 is /dev/ttyS6")
    assert "ekf2_gps_pos_x" in t and "gps" in t and "telem1" in t


def test_intents():
    assert intent_vector("Which parameter selects RC or joystick input?")["parameter"] == 1
    assert intent_vector("Show me the exploded view")["figure"] == 1


def test_every_handwritten_question_has_a_relevant_passage(pipe):
    qs = yaml.safe_load((ROOT / "data/questions_px4.yaml").read_text())
    assert len(qs) >= 70
    for q in qs:
        assert any(relevant(c, q) for c in pipe.r.chunks), q["q"]


def test_known_answers(pipe):
    r = pipe.ask("Which parameters set the GPS antenna position relative to the centre of gravity?")
    assert r["passages"][0]["section"] == "assembly/mount_gps_compass.md" or \
        any(p["section"] == "assembly/mount_gps_compass.md" for p in r["passages"][:5])
    r = pipe.ask("The EKF reports height estimate divergence. What is the most likely cause?")
    assert any("aliasing" in p["text"] for p in r["passages"][:5])


def test_figure_is_shown_when_asked(pipe):
    r = pipe.ask("Where is the debug port on the CUAV V5 nano?")
    assert r["images"] and "DSU7" in r["images"][0]["caption"]
