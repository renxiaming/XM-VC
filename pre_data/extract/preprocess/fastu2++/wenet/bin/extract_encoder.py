from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm
from wenet.dataset.dataset import Dataset
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.config import override_config

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--pad_num',
                        type=int,
                        default=-1,
                        help='num of padded zeros')
    parser.add_argument('--mode',
                        choices=[
                            'extract'
                        ],
                        default='extract',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--s_decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--s_num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--b_decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--b_num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--is_output_b_chunk',
                        action='store_true',
                        help='')
    parser.add_argument('--is_lh_output_chunk',
                        action='store_true',
                        help='')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_asr_model(configs)

    # Load dict
    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.decoder = torch.nn.ModuleList()

    model.eval()
    print(model)

    #encoder = model.encoder
    #encoder_ts = torch.jit.script(encoder)
    model.decoder = None
    model.ctc = None
    model.criterion_att = None
    model_ts = torch.jit.script(model)
    torch.jit.save(model_ts, "fastu2++.pt")
    model_quant = torch.ao.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)
    #self.script_model = torch.jit.script(vc)
    #torch.jit.save(self.script_model, os.path.join(output_dir, "vc.pt"))
    model_quant_ts = torch.jit.script(model_quant)
    torch.jit.save(model_quant_ts, "fastu2++.quant.pt")

    model.forward = model.forward_encoder_chunk
    input_names = ["fbanks", "offset", "required_cache_size", "att_cache", "cnn_cache"]
    output_names = ["asr_out", "r_att_cache", "r_cnn_cache"]
    inputs = (torch.rand(1, 23, 80), 10, 10, torch.rand(7, 4, 10, 128), torch.rand(7, 1, 256, 8))
    torch.onnx.export(model, inputs, "fastu2++.onnx", verbose=False, input_names=input_names, output_names=output_names, opset_version=11)
    import onnx, onnxruntime
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic("fastu2++.onnx", "fastu2++.quant.onnx", weight_type=QuantType.QUInt8)
    #model_ts

    os.makedirs(args.result_file, exist_ok=True)
    #exist_file = os.listdir(args.result_file)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_data_loader)):
            keys, feats, target, feats_lengths, target_lengths = batch
            #if os.path.isfile(os.path.join(args.result_file,  f"{keys[-1]}.npy")):
            #    continue
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)

            #feats = torch.from_numpy(np.load('feats1.npy')).to(device)
            np.save('feats.npy', feats.cpu().numpy(), allow_pickle=False)


            if args.mode == 'extract':
                #encoder_output = model.recognize(
                #    feats,
                #    feats_lengths,
                #    beam_size=args.beam_size,
                #    decoding_chunk_size=args.s_decoding_chunk_size,
                #    num_decoding_left_chunks=args.s_num_decoding_left_chunks,
                #    simulate_streaming=args.simulate_streaming,
                #    extract=True)
                # import pdb;pdb.set_trace()
                #encoder_out, _ = encoder.forward_chunk_by_chunk(
                #    feats,
                #    decoding_chunk_size=args.s_decoding_chunk_size,
                #    num_decoding_left_chunks=args.s_num_decoding_left_chunks,
                #)

                xs = feats
                decoding_chunk_size = args.s_decoding_chunk_size
                num_decoding_left_chunks=args.s_num_decoding_left_chunks
                subsampling = 4
                context = 7  # Add current frame
                stride = subsampling * decoding_chunk_size
                decoding_window = (decoding_chunk_size - 1) * subsampling + context
                num_frames = xs.size(1)
                att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
                cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
                outputs = []
                offset = 0
                required_cache_size = decoding_chunk_size * num_decoding_left_chunks

                # Feed forward overlap input step by step
                for cur in range(0, num_frames - context + 1, stride):
                    end = min(cur + decoding_window, num_frames)
                    chunk_xs = xs[:, cur:end, :]
                    #(y, att_cache, cnn_cache) = encoder_ts.forward_chunk(
                    (y, att_cache, cnn_cache) = model_ts.forward_encoder_chunk(
                        chunk_xs, offset, required_cache_size, att_cache, cnn_cache)
                    outputs.append(y)

                    offset += y.size(1)
                    #break
                ys = torch.cat(outputs, 1)
                encoder_out = ys

                encoder_out = encoder_out.transpose(1, 2)
                encoder_out = torch.nn.functional.interpolate(encoder_out, size=int(encoder_out.shape[2] * 16 / 5), mode='linear', align_corners=True)
                encoder_output = encoder_out.transpose(1, 2)

                encoder_output = encoder_output.data
                encoder_output = encoder_output.cpu().numpy()[0]
                utt = keys[-1]
                npy_path = os.path.join(args.result_file, '{}.npy'.format(utt))
                np.save(npy_path, encoder_output, allow_pickle=False)

            

if __name__ == '__main__':
    main()

