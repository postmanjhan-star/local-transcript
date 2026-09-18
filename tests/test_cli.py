from pathlib import Path
import subprocess
import sys

import yaml


def run_cli(*args):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, "-X", "utf8", str(root / "transcribe.py"), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", cwd=root)


def test_cli_help_and_errors(tmp_path):
    assert run_cli("--help").returncode == 0
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model": "tiny", "offline": True}), encoding="utf-8")
    missing = run_cli(tmp_path / "不存在.m4a", "--config", config)
    assert missing.returncode == 1 and "找不到輸入檔案" in missing.stdout
    empty = run_cli("--config", config)
    assert empty.returncode == 0 and "未找到錄音檔" in empty.stdout
    invalid = tmp_path / "壞掉的錄音.m4a"
    invalid.write_bytes(b"not audio")
    broken = run_cli(invalid, "--config", config)
    assert broken.returncode == 1 and "無法解析音訊格式" in broken.stdout
    assert "Traceback" not in broken.stdout and "Traceback" in next((tmp_path / "logs").glob("*.log")).read_text(encoding="utf-8")

