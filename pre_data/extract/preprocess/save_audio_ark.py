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

def write_kaldiio(data_path, ark_path, data_name, file_list):
    meta = [".".join(i.split(".")[:-1]) for i in file_list]
    os.makedirs(ark_path, exist_ok=True)
    numpy_array_dict = {}
    for i, info in enumerate(tqdm(meta)):
        #feat = np.load(os.path.join(data_path, info + ".npy"))
        feat, _ = librosa.load(os.path.join(data_path, info + ".wav"), sr=48000)
        numpy_array_dict[info] = np.squeeze(feat)
        if i % 10000 == 0:
            save_ark(os.path.join(ark_path, data_name), numpy_array_dict, append=True)
            numpy_array_dict.clear()
    save_ark(os.path.join(ark_path, data_name), numpy_array_dict, append=True)
    numpy_array_dict.clear()


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1]
    ark_path = sys.argv[2]
    data_name = sys.argv[3]
    file_list = sys.argv[4]
    file_list = open(file_list).read().splitlines()
    #print(data_path, ark_path, data_name, file_list)
    write_kaldiio(data_path, ark_path, data_name, file_list)