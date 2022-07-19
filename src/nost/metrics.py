import os
from typing import Union

import torch as th
from tqdm.auto import tqdm, trange

from src.nost.generation_config import GenerationRunParams, RunDirectory
from src.nost.compute_mauve_from_package import compute_mauve


class MetricsComputer:
    def __init__(
        self, run_directory: Union[str, RunDirectory],
    ):
        if isinstance(run_directory, str):
            run_directory = RunDirectory(path=run_directory)

        self.run_directory = run_directory

    def complete_metrics(self, seed):
        cm = {uid for uid, seed_ in self.run_directory.metrics if seed_ == seed}
        return cm.intersection(self.run_directory.complete_runs)

    def metrics_to_do(self, seed):
        return self.run_directory.complete_feats.difference(self.complete_metrics(seed))

    def do_remaining_metrics(self, seed, post_run_callback=None, **kwargs):
        for params in self.metrics_to_do():
            self.compute_metrics(params, seed=seed, post_run_callback=post_run_callback, **kwargs)

    def compute_metrics(self, params, seed, post_run_callback=None, **kwargs):
        p_feats = self.run_directory.load_groundtruth_feats(params)
        q_feats = self.run_directory.load_feats(params)

        metrics_obj = compute_mauve(p_features=p_feats, q_features=q_feats, seed=seed, **kwargs)

        def _handle(obj):
            if isinstance(obj, list) or isinstance(obj, float):
                return obj
            try:
                return obj.tolist()
            except:
                return float(obj)


        metrics_d = {}
        for k, v in metrics_obj.__dict__.items():
            metrics_d[k] = _handle(v)

        self.run_directory.record_metrics(params, seed, metrics_d)
        self.run_directory.save_metrics(params, seed, metrics_d)
