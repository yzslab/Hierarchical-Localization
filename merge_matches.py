import os
import sys
import h5py

from tqdm.auto import tqdm

path = sys.argv[1]

file_list = []
world_size_set = {}
for i in os.scandir(os.path.join(path, "match-chunks")):
    if not i.is_file():
        continue

    try:
        name, ext, world_size, global_rank = i.name.split(".")
    except:
        print("Skip {}".format(i.name))

    if global_rank > world_size:
        print("Skip {}".format(i.name))

    world_size_set[world_size] = world_size
    file_list.append(i.path)

assert len(world_size_set.keys()) == 1

output_path = os.path.join(path, "{}.h5".format(name))
# assert os.path.exists(output_path) is False, output_path
print(output_path)

with h5py.File(output_path, "a", libver="latest") as dst:
    for chunk in tqdm(file_list):
        with h5py.File(chunk, "r", libver="latest") as chunk_f:
            for _, v in tqdm(list(chunk_f.items()), leave=False):
                for _, src_grp in v.items():
                    dst_grp = dst.create_group(src_grp.name)
                    for name, data in src_grp.items():
                        dst_grp.create_dataset(name, data=data.__array__())
