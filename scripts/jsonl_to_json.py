import argparse
import json
import os
from typing import Dict


def write_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a JSONL file into per-record JSON files.")
    parser.add_argument("--input", required=True, help="Path to JSONL file")
    parser.add_argument("--output_dir", required=True, help="Directory to write JSON files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample_id = obj.get("id", f"gtl_{count:06d}")
            out_path = os.path.join(args.output_dir, f"{sample_id}.json")
            write_json(obj, out_path)
            count += 1

    print(f"Wrote {count} files to {args.output_dir}")


if __name__ == "__main__":
    main()
