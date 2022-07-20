import os
from typing import Union
from pprint import pformat
from functools import partial

import torch as th
from tqdm.auto import tqdm, trange
from tqdm.contrib.concurrent import process_map

from src.nost.generation_config import GenerationRunParams, GenerationRuns, RunDirectory
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

    def summarize_metrics(self, seed):
        gr = GenerationRuns(self.run_directory.complete_runs)
        vars, _ = gr.cv_form()
        for p, v in zip(gr.param_grid, vars):
            if (p, seed) in self.run_directory.metrics:
                print(v)
                print(f"\t{self.run_directory.metrics[(p, seed)]['mauve']}")

    def do_remaining_metrics(self, seed, post_run_callback=None, verbose=True, trialrun=False, n_concurrent=1, **kwargs):
        if n_concurrent > 1:
            to_do = list(self.metrics_to_do(seed))
            handler = partial(
                self.compute_metrics, seed=seed, post_run_callback=post_run_callback, verbose=verbose,
                trialrun=trialrun, **kwargs
            )
            process_map(handler, to_do, max_workers=n_concurrent)
        else:
            for params in self.metrics_to_do(seed):
                self.summarize_metrics(seed)
                self.compute_metrics(
                    params,
                    seed=seed,
                    post_run_callback=post_run_callback,
                    verbose=verbose,
                    trialrun=trialrun,
                    **kwargs
                )

    def compute_metrics(self, params, seed, post_run_callback=None, verbose=True, trialrun=False, **kwargs):
        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        vprint(f"computing metrics for {pformat(params.to_dict())}")

        p_feats = self.run_directory.load_groundtruth_feats(params)
        q_feats = self.run_directory.load_feats(params)

        metrics_obj = compute_mauve(p_features=p_feats, q_features=q_feats, seed=seed, verbose=verbose, **kwargs)

        vprint(f"mauve = {metrics_obj.mauve}")

        if trialrun:
            return

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

        if post_run_callback is not None:
            post_run_callback(self.run_directory, params, seed)
