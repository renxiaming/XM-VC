#!/usr/bin/env python3
import argparse
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package all folders starting with final_result_wavs under exp."
    )
    parser.add_argument(
        "--exp_dir",
        default="/home/work_nfs19/xmren/conversation_pipline/exp",
        help="Root exp directory to scan.",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Where to save tar.gz files. Default: current directory.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be packaged.",
    )
    return parser.parse_args()


def collect_target_folders(exp_dir: Path) -> list[Path]:
    # Collect dirs whose basename starts with final_result_wavs
    folders = {
        p for p in exp_dir.rglob("*")
        if p.is_dir() and p.name.startswith("final_result_wavs")
    }
    return sorted(folders, key=lambda p: str(p))


def package_folder(folder: Path, output_dir: Path) -> Path:
    # Use parent folder name as prefix to avoid filename collision
    prefix = folder.parent.name
    archive_name = f"{prefix}_{folder.name}.tar.gz"
    archive_path = output_dir / archive_name

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)

    return archive_path


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")

    folders = collect_target_folders(exp_dir)
    if not folders:
        print("No final_result_wavs* folders found.")
        return

    print(f"Found {len(folders)} folder(s).")
    for folder in folders:
        if args.dry_run:
            prefix = folder.parent.name
            archive_name = f"{prefix}_{folder.name}.tar.gz"
            print(f"[DRY_RUN] {folder} -> {output_dir / archive_name}")
            continue

        archive_path = package_folder(folder, output_dir)
        print(f"[OK] {folder} -> {archive_path}")


if __name__ == "__main__":
    main()