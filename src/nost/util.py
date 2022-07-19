from src.utils import load_and_tokenize_data


def load_ground_truth(enc, data_dir, prompt_source_file, min_len):
    segs = prompt_source_file.split('.')

    ds_name = segs[0]
    split = segs[1] if len(segs) > 2 else ''

    return load_and_tokenize_data(
        enc, data_dir, 1024, 10000, min_len=min_len, ds_name=ds_name, split=split
    )
