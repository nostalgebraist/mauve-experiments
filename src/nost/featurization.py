import os
from typing import Union, Optional
from pprint import pformat

import torch as th
from tqdm.auto import tqdm, trange

from src.nost.generation_config import GenerationRunParams, GenerationRuns, RunDirectory
from src.nost.compute_mauve_from_package import get_features_from_input
from src.nost.util import load_ground_truth, handle_bs_or_bs_map, apply_filters


def featurize_tokens(
    tokens, max_examples, batch_size, featurize_model_name='gpt2-large', device_id=0, name='',
    minimize_padding=True,
    minimize_padding_longest_first=True,
):
    tokens_ = tokens[:max_examples]
    tokens = tokens_

    if minimize_padding:
        token_length = [t.view(-1).shape[0] for t in tokens]

        sorter = sorted(list(range(len(token_length))), key=lambda i: token_length[i], reverse=minimize_padding_longest_first)

        unsorter = [None] * len(sorter)
        for i, ix in enumerate(sorter):
            unsorter[ix] = i

        assert None not in unsorter

        tokens = [tokens[ix] for ix in sorter]

    feat = get_features_from_input(
        features=None,
        tokenized_texts=tokens,
        texts=None,
        featurize_model_name=featurize_model_name,
        max_len=None,  # unused
        name=name,
        batch_size=batch_size,
        verbose=True,
        device_id=0
    )

    if minimize_padding:
        feat = feat[unsorter]

    return feat


class Featurizer:
    def __init__(
        self, run_directory: Union[str, RunDirectory], device_id=0, data_dir='data',
        featurize_model_name='gpt2-large',
    ):
        self.device_id = device_id
        self.data_dir = data_dir
        self.featurize_model_name = featurize_model_name

        if isinstance(run_directory, str):
            run_directory = RunDirectory(path=run_directory)

        self.run_directory = run_directory

    @property
    def complete_feats(self):
        return self.run_directory.complete_feats.intersection(self.run_directory.complete_runs)

    def feats_to_do(self, filters=None):
        to_do =  self.run_directory.complete_runs.difference(self.run_directory.complete_feats)

        n_before = len(to_do)
        to_do = apply_filters(to_do, filters)
        n_after = len(to_do)
        print(f"{n_after} to do after filters (vs {n_before} before)")
        return to_do

    def do_remaining_feats(
        self, bs_or_bs_map: Union[int, dict], post_run_callback=None, post_run_callback_groundtruth=None,
        minimize_padding=True, minimize_padding_longest_first=True,
        filters=None,
    ):
        for params in self.feats_to_do(filters):
            bs = handle_bs_or_bs_map(bs_or_bs_map, params.max_len)
            self.featurize_run(
                params,
                bs,
                post_run_callback=post_run_callback,
                post_run_callback_groundtruth=post_run_callback_groundtruth,
                minimize_padding=minimize_padding,
                minimize_padding_longest_first=minimize_padding_longest_first,
            )

    def featurize_run(
        self, params: GenerationRunParams, batch_size: int, post_run_callback=None, post_run_callback_groundtruth=None,
        minimize_padding=True, minimize_padding_longest_first=True,
    ):
        self.featurize_ground_truth(
            params,
            batch_size,
            post_run_callback_groundtruth,
            minimize_padding,
            minimize_padding_longest_first
        )  # skips internally if already done

        print(f"computing features for {pformat(params.to_dict())}")

        tokens = self.run_directory.load_tokens(params)

        feats = featurize_tokens(
            tokens,
            max_examples=params.max_num_generations,
            batch_size=batch_size,
            featurize_model_name=self.featurize_model_name,
            device_id=self.device_id,
            name=params.uid,
            minimize_padding=minimize_padding,
            minimize_padding_longest_first=minimize_padding_longest_first,
        )

        self.run_directory.save_feats(params, feats)
        self.run_directory.record_feats(params)

        if post_run_callback is not None:
            post_run_callback(self.run_directory, params)

    def featurize_ground_truth(
        self, params: GenerationRunParams, batch_size: int, post_run_callback=None,
        minimize_padding=True, minimize_padding_longest_first=True,
    ):
        if os.path.exists(self.run_directory.ground_truth_feats_path(params)):
            return

        print(f"computing features for {params.prompt_source_file}")

        tokens = load_ground_truth(
            enc=None,  # should be pre-tokenized
            data_dir=self.data_dir,
            prompt_source_file=params.prompt_source_file,
            min_len=params.prompt_len
        )

        tokens = [seq[:, :params.max_len] for seq in tokens]
        print(f"max token len: {max(seq.shape[1] for seq in tokens)}")

        feats = featurize_tokens(
            tokens,
            max_examples=params.max_num_generations,
            batch_size=batch_size,
            featurize_model_name=self.featurize_model_name,
            device_id=self.device_id,
            name=params.prompt_source_file,
            minimize_padding=minimize_padding,
            minimize_padding_longest_first=minimize_padding_longest_first,
        )

        self.run_directory.save_groundtruth_feats(params, feats)

        if post_run_callback is not None:
            post_run_callback(self.run_directory, params)
