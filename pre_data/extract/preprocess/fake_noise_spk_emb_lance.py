from aslp_utils import LanceWriter, LanceReader, AudioData, FloatNPYData
import sys
from tqdm import tqdm

input_dir = sys.argv[1]
output_dir = sys.argv[2]

reader = LanceReader(input_dir, target_cls=FloatNPYData)
writer = LanceWriter(output_dir, target_cls=FloatNPYData)

ids = reader.get_ids()
out_data = []
WRITE_INTERVAL = 100000
for row in tqdm(range(len(ids))):
    data = reader.get_datas_by_rows([row])[0]
    out_data.append(FloatNPYData(data_id=data.data_id+'_noise', data=data.data))
    if len(out_data) > WRITE_INTERVAL:
        writer.write_parallel(out_data)
        out_data = []
writer.write_parallel(out_data)
