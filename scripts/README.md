# Demo GIF Generation

This folder contains `generate_demo_gif.py`, a Playwright-based recorder that
captures a short demo of the Streamlit dashboard and assembles it into `demo.gif`.

Prerequisites
-------------
- Python 3.11+
- Playwright: `pip install playwright` then `playwright install`
- imageio (optional): `pip install imageio imageio-ffmpeg`
- Alternatively, `ffmpeg` to assemble frames manually

Usage
-----
1. Start the Streamlit dashboard locally (default URL `http://localhost:8501`).
2. Run the script:

```bash
python scripts/generate_demo_gif.py
```

If `imageio`/`imageio-ffmpeg` are unavailable, the script will still capture
frames to `reports/demo_frames/`. You can assemble them using `ffmpeg`:

```bash
ffmpeg -framerate 10 -i reports/demo_frames/frame_%03d.png -pix_fmt rgb24 -vf "scale=iw:-1" demo-hd.mp4
ffmpeg -i demo-hd.mp4 -vf "fps=15,scale=640:-1:flags=lanczos" -loop 0 demo.gif
```

Notes
-----
This script is a pragmatic capture tool for creating short demo GIFs. For
production-ready recordings prefer Playwright's built-in video recording or
a desktop screen recorder for highest fidelity.
