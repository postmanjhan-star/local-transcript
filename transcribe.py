"""Source-checkout entry point; also usable without editable installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from meeting_transcriber.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

