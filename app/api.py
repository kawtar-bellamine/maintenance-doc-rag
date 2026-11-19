"""FastAPI service: POST /api/ask returns the answer, the passages and the figures; the React UI is served at /."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docrag.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="PX4 documentation assistant")
pipe = Pipeline(ROOT / "data/index")


class Ask(BaseModel):
    question: str
    with_images: bool = True


@app.post("/api/ask")
def ask(body: Ask):
    return pipe.ask(body.question, with_images=body.with_images)


@app.get("/api/health")
def health():
    return {"chunks": len(pipe.r.chunks), "figures": len(pipe.r.figures)}


dist = ROOT / "web" / "dist"
if dist.exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(dist / "index.html")
