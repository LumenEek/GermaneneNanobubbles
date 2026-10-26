"""Choose a local figure export directory, with optional explicit external export."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURE_DIR = Path(os.environ.get('HB_FIGURE_DIR', ROOT / 'results/manuscript_figures')).expanduser().resolve()


def figure_destination(filename):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR / filename
