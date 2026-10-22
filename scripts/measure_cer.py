"""Zero-dependency, fully offline character error rate (CER) evaluator."""
import argparse
import json
from pathlib import Path
import unicodedata


def read_transcript(path, raw=False):
    data = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        document = json.loads(data)
        segments = document["segments"]
        return "\n".join(str(s.get("raw_text" if raw else "text", s.get("text", ""))) for s in segments)
    return data


def canonicalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(c for c in value if not c.isspace() and not unicodedata.category(c).startswith("P"))


def edit_distance(a, b):
    if len(b) > len(a):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ac in enumerate(a, 1):
        current = [i]
        for j, bc in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1,
                               previous[j-1] + (ac != bc)))
        previous = current
    return previous[-1]


def measure(reference, prediction):
    a, b = canonicalize(reference), canonicalize(prediction)
    if not a:
        raise ValueError("參考逐字稿為空白")
    if max(len(a), len(b)) > 8000:
        raise ValueError("樣本過長，請截取 1–5 分鐘的錄音")
    distance = edit_distance(a, b)
    return {"reference_characters": len(a), "prediction_characters": len(b),
            "edit_distance": distance, "cer": distance / len(a),
            "cer_percent": round(distance * 100 / len(a), 2)}


def main():
    parser = argparse.ArgumentParser(description="免費本地逐字稿 CER 評估")
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--raw", action="store_true", help="使用 JSON 中的原始 ASR 文字")
    args = parser.parse_args()
    try:
        result = measure(read_transcript(args.reference), read_transcript(args.prediction, args.raw))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"CER 評估失敗：{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
