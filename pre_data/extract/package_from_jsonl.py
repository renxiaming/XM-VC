#!/usr/bin/env python3
import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a jsonl file, collect final_result_wavs folders from wav_path, "
            "and package each folder into a tar.gz archive."
        )
    )
    parser.add_argument("--jsonl", required=True, help="Input jsonl file path.")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save generated tar.gz files. Default: current directory.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print folders to be packaged, do not create archives.",
    )
    return parser.parse_args()


def extract_folder_from_wav_path(wav_path: str) -> Optional[Path]:
    path = Path(wav_path)
    parts = path.parts
    if "final_result_wavs" not in parts:
        return None
    idx = parts.index("final_result_wavs")
    return Path(*parts[: idx + 1])


def collect_target_folders(jsonl_path: Path) -> Dict[Path, str]:
    folder_to_prefix: Dict[Path, str] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skip invalid JSON at line {line_no}")
                continue

            wav_path = item.get("wav_path")
            if not wav_path:
                continue

            folder = extract_folder_from_wav_path(wav_path)
            if folder is None:
                print(f"[WARN] No final_result_wavs in wav_path (line {line_no}): {wav_path}")
                continue

            # Example:
            # /.../dialogue/yuzhoufm/yuzhoufm_0/final_result_wavs
            # prefix = yuzhoufm_0
            prefix = folder.parent.name
            folder_to_prefix[folder] = prefix

    return folder_to_prefix


def package_folder(folder: Path, prefix: str, output_dir: Path) -> Path:
    archive_name = f"{prefix}_{folder.name}.tar.gz"
    archive_path = output_dir / archive_name

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)

    return archive_path


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl file not found: {jsonl_path}")

    folder_to_prefix = collect_target_folders(jsonl_path)

    if not folder_to_prefix:
        print("No valid final_result_wavs folders found from jsonl.")
        return

    print(f"Found {len(folder_to_prefix)} unique folder(s) to package.")

    for folder, prefix in sorted(folder_to_prefix.items(), key=lambda x: str(x[0])):
        if not folder.exists():
            print(f"[WARN] Folder not found, skip: {folder}")
            continue

        if args.dry_run:
            print(f"[DRY_RUN] {folder} -> {output_dir / (prefix + '_' + folder.name + '.tar.gz')}")
            continue

        archive_path = package_folder(folder, prefix, output_dir)
        print(f"[OK] {folder} -> {archive_path}")


if __name__ == "__main__":
    main()
