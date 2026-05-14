import os
import sys
import signal
import json
import argparse
from pathlib import Path
import torchaudio
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

SPEECH_TOKEN_DIR = "/home/node57_data/hkxie/4O/streaming_fm/data/s3token2"
BATCH_SIZE = 5000

def read_audio_speech_pairs(txt_file_path):
    pairs = []
    with open(txt_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 3:
                continue
            utt, wav_path, duration = parts[:3]
            speech_token_path = os.path.join(SPEECH_TOKEN_DIR, f"{utt}.hubert_code.npy")
            pairs.append((wav_path, speech_token_path, float(duration)))
    return pairs

def prepare_txt_wavs_dir(input_txt_file):
    audio_pairs = read_audio_speech_pairs(input_txt_file)
    results = [
        {"audio_path": wav, "speech_token_path": token, "duration": duration}
        for wav, token, duration in audio_pairs
    ]
    durations = [item["duration"] for item in results]
    return results, durations

def save_prepped_dataset(out_dir, result, duration_list):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=BATCH_SIZE) as writer:
        for line in tqdm(result, desc="写入 raw.arrow"):
            writer.write(line)
    with open(out_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

def prepare_and_save_set(inp_file, out_dir):
    result, durations = prepare_txt_wavs_dir(inp_file)
    save_prepped_dataset(out_dir, result, durations)

def cli():
    parser = argparse.ArgumentParser(description="根据 txt 文件生成训练所需的数据集文件")
    parser.add_argument("inp_file", type=str, help="包含 utt wav_path duration text 的 txt 文件路径")
    parser.add_argument("out_dir", type=str, help="输出目录")
    args = parser.parse_args()
    prepare_and_save_set(args.inp_file, args.out_dir)

if __name__ == "__main__":
    cli()

