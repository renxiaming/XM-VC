import os
import json
import sys

dirs = sys.argv[1]

wavs = os.listdir(dirs)

data = []

for wav in wavs:
	s = {}
	s["wav"] = f"{dirs}/{wav}"#f"/home/work_nfs6/ypjiang/data/cbhgar/wavs/{wav}"
	s["key"] = ".".join(wav.split(".")[:-1])
	s["txt"] = ""
	data.append(json.dumps(s))
with open(f"data.list", "w") as f:
	f.write("\n".join(data))
