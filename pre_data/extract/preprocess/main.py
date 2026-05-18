import argparse
from extractors import MelExtractor, Lf0Extractor, EnergyExtractor, MelLf0UVEnergyExtractor, MelLanceExtractor
from config import MelABSValue
from config import preConfiged16K, preConfiged16K_10ms, preConfiged22K, preConfiged24K, preconfiged24K256, preConfiged44K, preConfiged48K

def main(args):
    wav_dir = args.wav_dir
    out_dir = args.out_dir
    sr = args.sr
    jobs = args.jobs
    inc = args.inc
    for_vc = args.vc
    use_lance = args.use_lance
    filelist = args.filelist

    if sr == "16k":
        config = preConfiged16K
    elif sr == "16k10hs":
        config = preConfiged16K_10ms
    elif sr == "22k":
        config = preConfiged22K
    elif sr == "24k":
        config = preConfiged24K
    elif sr == "24k256":
        config = preconfiged24K256
    elif sr == "44k":
        config = preConfiged44K
    elif sr == "48k":
        config = preConfiged48K
    else:
        raise Exception(f"Unknown preconfiged sample rate {sr}")
    
    if for_vc:
        config.max_abs_value = MelABSValue.VC
    print(config)

    if not use_lance:
        extractor = MelExtractor(config, jobs, inc)
        #extractor = Lf0Extractor(config, jobs, inc)
        #extractor = EnergyExtractor(config, jobs, inc)
        #extractor = MelLf0UVEnergyExtractor(config, jobs, inc)
        extractor(wav_dir, out_dir)
    else:
        extractor = MelLanceExtractor(config, jobs, inc)
        extractor(wav_dir, out_dir, filelist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_dir", type=str, help="wav dir")
    parser.add_argument("out_dir", type=str, help="output dir")
    parser.add_argument("-s", "--sr", choices=["16k","16k10hs", "22k", "24k", "24k256", "44k", "48k"], type=str, default="16k", help="preconfiged sample rate config")
    parser.add_argument("-j", "--jobs", type=int, default=40)
    parser.add_argument("-i", "--inc", action="store_true", default=False)
    parser.add_argument("--vc", action="store_true", default=False)
    parser.add_argument("--use_lance", type=bool, default=False)
    parser.add_argument("--filelist", type=str, default=None)
    args = parser.parse_args()
    main(args)
