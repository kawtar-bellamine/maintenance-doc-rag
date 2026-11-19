"""Fetch the real corpus: the PX4 Autopilot User Guide (Dronecode Foundation, CC BY 4.0), pinned to one commit so
the evaluation is reproducible. Only the English pages used here and the images they reference are downloaded."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = "https://github.com/PX4/PX4-user_guide.git"
COMMIT = "2fa62338099f18fc3092197e856615eca94d2975"        # 18 April 2026
SECTIONS = ["en/assembly", "en/flight_controller", "en/config", "en/getting_started", "en/flying",
            "en/advanced_config"]
EXCLUDE = {"advanced_config/parameter_reference.md"}        # 49,000 generated lines of parameter tables


def fetch(dest: Path) -> Path:
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *SECTIONS], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", COMMIT], check=True, capture_output=True)
    dirs = set()
    for f in (dest / "en").rglob("*.md"):
        for p in re.findall(r"assets/[^) \"']+\.(?:png|jpg|jpeg|gif)", f.read_text(errors="ignore")):
            dirs.add(p.rsplit("/", 1)[0])
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "add", *sorted(dirs)], check=True)
    return dest


if __name__ == "__main__":
    print(fetch(Path("data/px4")))
