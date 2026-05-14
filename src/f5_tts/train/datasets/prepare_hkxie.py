import os
import sys
import signal
import subprocess
import shutil
import concurrent.futures
import multiprocessing
from contextlib import contextmanager

sys.path.append(os.getcwd())

import argparse
import csv
import json
from pathlib import Path

import torchaudio
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

# 常量：speech token 文件所在目录
SPEECH_TOKEN_DIR = "/home/node57_data/hkxie/4O/streaming_fm/data/s3token2"

# 配置常量
BATCH_SIZE = 5000         # 文本转换批次大小（本例中不再需要转换，但保持并行任务批次大小）
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 10)  # 保留一个 CPU 核心
THREAD_NAME_PREFIX = "AudioProcessor"
CHUNK_SIZE = 5000         # 每个 worker 处理的文件数量

executor = None  # 全局 executor，用于清理

@contextmanager
def graceful_exit():
    """用于优雅退出，响应 SIGINT/SIGTERM 信号"""
    def signal_handler(signum, frame):
        print("\n收到退出信号，正在清理...")
        if executor is not None:
            print("正在关闭 executor ...")
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)

def read_audio_speech_pairs(txt_file_path):
    """
    从 txt 文件中读取音频和 speech token 信息。
    每行格式：utt wav_path duration text
    其中：
      - utt：语音样本标识符（用于构造 speech token 文件名）
      - wav_path：音频文件的绝对路径
      - duration：音频时长（秒），字符串形式，需要转换为 float
      - text：原始文本（本脚本中不再使用）
    返回列表，每个元素为 (wav_path, speech_token_path, duration)
    """
    pairs = []
    txt_file = Path(txt_file_path)
    with open(txt_file, mode="r", encoding="utf-8") as f:
        for line in f:
            # 按最大分割数 3 分割，确保 text 部分整体为第四个字段（但我们不使用它）
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 3:
                continue  # 格式错误则跳过
            utt = parts[0]
            wav_path = parts[1]
            try:
                duration = float(parts[2])
            except ValueError:
                print(f"无法转换时长：{parts[2]}，跳过 {utt}")
                continue
            # 构造 speech token 文件路径，假设命名为 utt.npy
            speech_token_path = os.path.join(SPEECH_TOKEN_DIR, f"{utt}.hubert_code.npy")
            pairs.append((wav_path, speech_token_path, duration))
    return pairs

def process_audio_file_txt(wav_path, speech_token_path, provided_duration):
    """
    处理单个音频文件：
      - 检查音频文件是否存在
      - 使用 txt 中提供的时长，并判断其是否为正数
      - 返回 (wav_path, speech_token_path, duration)
    """
    if not Path(wav_path).exists():
        print(f"音频文件 {wav_path} 不存在，跳过")
        return None
    try:
        duration = float(provided_duration)
        if duration <= 0:
            raise ValueError(f"时长 {duration} 非正")
        # 可选：你也可以选择额外检查 speech token 文件是否存在（这里不强制）
        return (wav_path, speech_token_path, duration)
    except Exception as e:
        print(f"处理 {wav_path} 时出错：{e}，跳过")
        return None

def prepare_txt_wavs_dir(input_txt_file, num_workers=None):
    """
    读取 txt 文件，处理所有音频文件，返回处理后的样本列表、时长列表。
    每个样本为 (wav_path, speech_token_path, duration)
    """
    audio_pairs = read_audio_speech_pairs(input_txt_file)
    total_files = len(audio_pairs)
    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, total_files)
    print(f"\n使用 {worker_count} 个线程处理 {total_files} 个音频文件...")

    global executor
    results = []
    with graceful_exit():
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix=THREAD_NAME_PREFIX
        ) as exec:
            executor = exec
            for i in range(0, total_files, CHUNK_SIZE):
                chunk = audio_pairs[i : i + CHUNK_SIZE]
                chunk_futures = [
                    executor.submit(process_audio_file_txt, pair[0], pair[1], pair[2])
                    for pair in chunk
                ]
                for future in tqdm(
                    chunk_futures,
                    total=len(chunk),
                    desc=f"处理第 {i//CHUNK_SIZE + 1} 批，共 { (total_files+CHUNK_SIZE-1)//CHUNK_SIZE} 批"
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"处理文件时发生错误：{e}")
            executor = None

    processed = [res for res in results if res is not None]
    if not processed:
        raise RuntimeError("没有有效的音频文件被处理！")

    # 提取时长列表
    durations = [res[2] for res in processed]

    # 构造最终结果列表，每个样本为字典，包含音频路径、speech token 路径和时长
    final_results = []
    for wav_path, speech_token_path, duration in processed:
        final_results.append({
            "audio_path": wav_path,
            "speech_token_path": speech_token_path,
            "duration": duration,
        })

    return final_results, durations

def save_prepped_dataset(out_dir, result, duration_list):
    """
    保存预处理后的数据：
      - raw.arrow：每条记录包含 audio_path, speech_token_path, duration
      - duration.json：保存所有时长信息
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n保存到 {out_dir} ...")

    # 保存 Arrow 格式数据
    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=5000) as writer:
        for line in tqdm(result, desc="写入 raw.arrow ..."):
            writer.write(line)

    # 保存时长 JSON 文件
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    dataset_name = out_dir.stem
    print(f"\n数据集 {dataset_name} 样本数量: {len(result)}")
    print(f"总时长: {sum(duration_list)/3600:.2f} 小时")

def prepare_and_save_set(inp_file, out_dir, num_workers: int = None):
    result, durations = prepare_txt_wavs_dir(inp_file, num_workers=num_workers)
    save_prepped_dataset(out_dir, result, durations)

def cli():
    try:
        parser = argparse.ArgumentParser(
            description="根据 txt 文件生成训练所需的数据集文件",
            epilog="""
Examples:
    python prepare_txt_wavs.py /path/to/your/input.txt /path/to/output_dir --workers 4
            """
        )
        parser.add_argument("inp_file", type=str, help="包含 utt wav_path duration text 的 txt 文件路径")
        parser.add_argument("out_dir", type=str, help="输出目录，用于保存 raw.arrow 和 duration.json")
        parser.add_argument("--workers", type=int, help=f"线程数 (默认: {MAX_WORKERS})")
        args = parser.parse_args()
        
        prepare_and_save_set(args.inp_file, args.out_dir, num_workers=args.workers)
        
        # python prepare_hkxie.py /home/work_nfs14/code/hkxie/ASR/understanding_LLM_task/datalist/1whdata.txt /home/work_nfs14/code/hkxie/TTS/F5-TTS/data --workers 16

    except KeyboardInterrupt:
        print("\n用户取消操作，正在清理...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
