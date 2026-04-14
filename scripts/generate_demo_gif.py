#!/usr/bin/env python3
"""Generate a demo GIF by automating the Streamlit dashboard with Playwright.

This script requires Playwright and imageio-ffmpeg (or ffmpeg CLI) to assemble
frames into a GIF. It runs the browser headless and captures a guided sequence
of screenshots while the Streamlit dashboard is running.
"""
from __future__ import annotations

import time
import os
from pathlib import Path
from typing import List

try:
    from playwright.sync_api import sync_playwright
except Exception as exc:
    raise RuntimeError(
        "playwright is required to run this script. Install with `pip install playwright` and run `playwright install`."
    ) from exc

try:
    import imageio
except Exception:
    imageio = None

OUT_DIR = Path("reports/demo_frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = os.getenv("DEMO_URL", "http://localhost:8501")


def capture_sequence(output_gif: str = "demo.gif", duration_s: int = 30):
    frames: List[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(URL)
        time.sleep(2)

        # 0–5s: ensure app loaded and show sidebar status
        page.screenshot(path=str(OUT_DIR / "frame_000.png"))
        frames.append(str(OUT_DIR / "frame_000.png"))

        # 5–12s: click start simulation button
        try:
            btn = page.locator('button:has-text("▶  Run Simulation")')
            if btn.count() == 0:
                btn = page.locator('button:has-text("Run Simulation")')
            btn.first.click()
        except Exception:
            # fallback: try generic button
            els = page.locator("button")
            if els.count() > 0:
                els.nth(0).click()

        for i in range(1, 10):
            time.sleep(0.8)
            pth = OUT_DIR / f"frame_{i:03d}.png"
            page.screenshot(path=str(pth))
            frames.append(str(pth))

        # 12–20s: wait for first REM entry / ACh spike - poll for hypnogram canvas
        for i in range(10, 20):
            time.sleep(0.6)
            pth = OUT_DIR / f"frame_{i:03d}.png"
            page.screenshot(path=str(pth))
            frames.append(str(pth))

        # 20–26s: capture narrative panel
        for i in range(20, 26):
            time.sleep(0.5)
            pth = OUT_DIR / f"frame_{i:03d}.png"
            page.screenshot(path=str(pth))
            frames.append(str(pth))

        # 26–30s: final overview
        for i in range(26, 30):
            time.sleep(0.5)
            pth = OUT_DIR / f"frame_{i:03d}.png"
            page.screenshot(path=str(pth))
            frames.append(str(pth))

        browser.close()

    if imageio is not None:
        imgs = [imageio.imread(fp) for fp in frames]
        imageio.mimsave(output_gif, imgs, fps=10)
        print(f"Wrote GIF: {output_gif}")
    else:
        print("Captured frames to:", frames)
        print(
            "Install imageio and ffmpeg to assemble GIF, or use the included README for ffmpeg commands."
        )


if __name__ == "__main__":
    capture_sequence()
