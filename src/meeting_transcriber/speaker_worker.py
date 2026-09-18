"""Private subprocess entry point; never imports the Whisper/ORT Python runtime."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

from .config import load_config
from .diarization import Diarizer
from .utils import read_json, write_json


def main() -> int:
    job = Path(sys.argv[1]).resolve()
    try:
        request = read_json(job / "request.json")
        # JSON is a YAML subset; reuse the full configuration validation path.
        write_json(job / "config.json", request["config"])
        config = load_config(job / "config.json")
        def emit(value: dict) -> None:
            print(json.dumps(value, ensure_ascii=False), flush=True)
        engine = Diarizer(config, lambda text: emit({"type": "message", "text": text}),
                         lambda event: emit({"type": "progress", "event": asdict(event)}),
                         lambda: (job / "cancel").exists())
        result = engine._run_local(Path(request["source"]), Path(request["folder"]), request["segments"])
        write_json(job / "result.json", {"segments": result})
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        write_json(job / "error.json", {"message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

