import gc, time
from typing import Union
from pprint import pformat

import torch as th
from tqdm.auto import tqdm, trange
from transformers import LogitsProcessorList

from src.nost.generation_config import GenerationRunParams, GenerationRuns, RunMetadata, RunDirectory
from src.nost.util import load_ground_truth, handle_bs_or_bs_map
from src.utils import get_model_and_tokenizer

from src.decoding_methods import BreakrunsLogitsProcessor, MirostatLogitsProcessor


def make_override_get_breakruns(base_temperature, tau, tokenizer=None, debug=False,
                                ):
    def _override_get_breakruns(*args, **kwargs) -> LogitsProcessorList:
        if debug:
            print('logits processor call')
        processors = [
            BreakrunsLogitsProcessor(
                base_temperature=base_temperature,
                tau=tau,
                tokenizer=tokenizer,
                debug=debug
            )
        ]
        return LogitsProcessorList(processors)
    return _override_get_breakruns


def make_override_get_mirostat(tau, n=50000, learning_rate=1, debug=False):
    def _override_get_mirostat(*args, **kwargs) -> LogitsProcessorList:
        if debug:
            print('logits processor call')
        processors = [
            MirostatLogitsProcessor(
                tau=tau, n=n, learning_rate=learning_rate,
            )
        ]
        return LogitsProcessorList(processors)
    return _override_get_mirostat



class GenerationRunner:
    def __init__(
        self, run_directory: Union[str, RunDirectory], runs: GenerationRuns, device='cuda:0', data_dir='data',
        model_after_load_hook=None,
    ):
        self.runs = runs
        self.device = device
        self.data_dir = data_dir
        self.model_after_load_hook = model_after_load_hook

        if isinstance(run_directory, str):
            run_directory = RunDirectory(path=run_directory)

        self.run_directory = run_directory

        self._model_name = None
        self._model = None
        self._enc = None
        self._orig_logits_processor = None

        self._prompt_source_file = None
        self._prompt_data = None

    @property
    def complete_runs(self):
        return self.run_directory.complete_runs.intersection(self.runs.param_grid)

    def runs_to_do(self):
        return [r for r in self.runs.param_grid if r not in self.complete_runs]

    def cleanup(self):
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

    def _set_model(self, model_name):
        if self._model_name != model_name:
            self._model = None
            self.cleanup()

            self._model, self._enc = get_model_and_tokenizer(model_name, self.device, self.model_after_load_hook)
            self._model.requires_grad_(False)

            self._orig_get_logits_processor = self._model._get_logits_processor
            self._model_name = model_name

    def _set_data(self, prompt_source_file, prompt_len):
        if self._prompt_source_file != prompt_source_file:
            segs = prompt_source_file.split('.')

            ds_name = segs[0]
            split = segs[1] if len(segs) > 2 else ''

            self._prompt_data = load_ground_truth(self._enc, self.data_dir, prompt_source_file, prompt_len)

            self._prompt_source_file = prompt_source_file

    def do_remaining_runs(self, bs_or_bs_map: Union[int, dict], debug=False, post_run_callback=None):
        for params in self.runs_to_do():
            bs = handle_bs_or_bs_map(bs_or_bs_map, params.max_len)
            self._do_run(params, bs, debug=debug, post_run_callback=post_run_callback)

    def _do_run(self, params: GenerationRunParams, bs: int, debug=False, post_run_callback=None):
        print(f"generating for {pformat(params.to_dict())}")

        self._set_model(params.model_name)
        self._set_data(params.prompt_source_file, params.prompt_len)

        using_breakruns = params.breakruns_tau > 0
        using_mirostat = params.mirostat_tau is not None and params.mirostat_tau > 0

        if using_breakruns:
            self._model._get_logits_processor = make_override_get_breakruns(
                base_temperature=params.temperature,
                tau=params.breakruns_tau,
                tokenizer=self._enc,
                debug=debug
            )

            params_effective = params.replace(temperature=1.0)
        elif using_mirostat:
            self._model._get_logits_processor = make_override_get_mirostat(
                tau=params.mirostat_tau,
            )

            params_effective = params
        else:
            self._model._get_logits_processor = self._orig_get_logits_processor
            params_effective = params

        th.manual_seed(params.seed)

        outs = []
        have = len(outs)

        prompt_data = self._prompt_data[params.prompt_source_offset:]

        if len(prompt_data) < params.max_num_generations:
            raise ValueError(f"{len(prompt_data)} data vs {params.max_num_generations} max_num_generations")

        t1 = time.time()

        for i in trange(params.max_num_generations // bs + int(params.max_num_generations % bs != 0)):
            offset_next = min(have+bs, params.max_num_generations)
            b = th.cat([t[:, :params.prompt_len] for t in prompt_data[have:offset_next]]).to(self.device)

            typical_p = None
            if 0 < params.typical_decoding_tau < 1:
                typical_p = params.typical_decoding_tau

            out = self._model.generate(
                b,
                do_sample=True,
                use_cache=True,
                eos_token_id=50256,
                pad_token_id=50256,
                max_length=params.max_len,
                temperature=params_effective.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                typical_p=typical_p,
            )

            outs.extend([s[s != 50256] for s in out.cpu()])
            if debug and i == 0:
                print([self._enc.decode(seq) for seq in outs])
            have = len(outs)

        delta = time.time() - t1
        meta = RunMetadata(runtime_seconds=delta, batch_size=bs)

        self.run_directory.save_tokens(params, outs)

        self.run_directory.record(params, meta, writefile=True)

        if post_run_callback is not None:
            post_run_callback(self.run_directory, params)
