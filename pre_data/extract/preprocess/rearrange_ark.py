import os
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
from functools import partial
import kaldiio


def save_ark(name, numpy_array_dict, append=False):
    ark_name = "{}.ark".format(name)
    scp_name = "{}.scp".format(name)
    kaldiio.save_ark(ark_name, numpy_array_dict, scp=scp_name, append=append)

def write_kaldiio(ark_path, scp_path, ref_scp_path, data_name):
    ark = {}
    numpy_array_dict = {}
    for line in open(scp_path).read().splitlines():
        filename, offset = line.split(" ")
        ark[filename] = offset
    for i, line in tqdm(enumerate(open(ref_scp_path).read().splitlines())):
        filename, _ = line.split(" ")
        offset = ark[filename]
        feat = kaldiio.load_mat(offset)
        numpy_array_dict[filename] = np.squeeze(feat)
        if i % 10000 == 0:
            save_ark(os.path.join(ark_path, data_name), numpy_array_dict, append=True)
            numpy_array_dict.clear()
    save_ark(os.path.join(ark_path, data_name), numpy_array_dict, append=True)
    numpy_array_dict.clear()


if __name__ == "__main__":
    import sys

    ark_path = sys.argv[1]
    scp_path = sys.argv[2]
    ref_scp_path = sys.argv[3]
    data_name = sys.argv[4]
    #print(data_path, ark_path, data_name, file_list)
    write_kaldiio(ark_path, scp_path, ref_scp_path, data_name)