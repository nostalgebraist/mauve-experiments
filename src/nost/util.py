from typing import Union

from src.utils import load_and_tokenize_data


def load_ground_truth(enc, data_dir, prompt_source_file, min_len):
    segs = prompt_source_file.split('.')

    ds_name = segs[0]
    split = segs[1] if len(segs) > 2 else ''

    eos_prepend = False
    if ds_name.endswith('_eos'):
        eos_prepend = True
        ds_name = ds_name[:-len('_eos')]

    return load_and_tokenize_data(
        enc, data_dir, 1024, 10000, min_len=min_len, ds_name=ds_name, split=split,
        eos_prepend=eos_prepend
    )



def handle_bs_or_bs_map(bs_or_bs_map: Union[int, dict], max_len: int):
    bs = None

    if isinstance(bs_or_bs_map, int):
        bs = bs_or_bs_map
    elif isinstance(bs_or_bs_map, dict):
        if max_len in bs_or_bs_map:
            bs = bs_or_bs_map[max_len]
        else:
            for val in sorted(bs_or_bs_map.keys(), reverse=True):
                if val < max_len:
                    bs = val
                    break
        print(f'Using bs={bs} for L={max_len}')
    else:
        raise TypeError(type(bs_or_bs_map))
    return bs
