import os
import json
import hashlib

from pprint import pformat
from io import StringIO
from itertools import product, chain
from dataclasses import dataclass, asdict, astuple, fields, replace
from typing import Optional, TypeVar, Type, Tuple

T = TypeVar("T")


def _indent(s, tab='\t'):
    return tab + s.replace('\n', '\n' + tab)


class DictJsonMixin:
    def to_dict(self) -> dict:
        return NotImplemented

    @classmethod
    def from_dict(cls: Type[T], d: dict) -> T:
        return cls(**d)

    def to_json_file(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json_file(cls: Type[T], path: str) -> T:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)

        return cls.from_dict(d)


class DataclassHelpersMixin(DictJsonMixin):
    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self: T, **changes) -> T:
        return replace(self, **changes)

    @classmethod
    def field_names(cls) -> Tuple[str]:
        return tuple(fd.name for fd in fields(cls))

    def _to_tuple(self) -> tuple:
        return tuple((name, getattr(self, name, None)) for name in self.field_names())

    @property
    def uid(self) -> str:
        u_str = repr(self._to_tuple())
        return hashlib.md5(u_str.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class GenerationRunParams(DataclassHelpersMixin):
    model_name: str

    seed: int
    max_len: int = 1024
    max_num_generations: int = 5000

    prompt_len: int = 35
    prompt_source_file: str = 'webtext.test.jsonl'
    prompt_source_offset: int = 0

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    breakruns_tau: float = 0.0
    typical_decoding_tau: float = 0.0
    mirostat_tau: Optional[float] = None


@dataclass
class RunMetadata(DataclassHelpersMixin):
    runtime_seconds: float
    batch_size: int


class GenerationRuns(DictJsonMixin):
    def __init__(self, param_grid):
        self.param_grid = param_grid

    @staticmethod
    def from_param_options(**params_or_param_options):
        param_options = {}
        for name, val in params_or_param_options.items():
            param_options[name] = val if isinstance(val, list) or isinstance(val, tuple) else (val,)

        param_grid = tuple()

        for vs in product(*param_options.values()):
            params = {name: v for name, v in zip(param_options.keys(), vs)}
            param_grid += (GenerationRunParams(**params),)

        return GenerationRuns(param_grid)

    @property
    def constants(self):
        def uniques(name):
            return set(getattr(ps, name) for ps in self.param_grid)

        constants = {}
        for name in GenerationRunParams.field_names():
            u = uniques(name)
            if len(u) == 1:
                constants[name] = list(u)[0]

        return constants

    def cv_form(self):
        constants = self.constants

        var_grid = tuple()

        for params in self.param_grid:
            var_grid += ({k: v for k, v in params.to_dict().items() if k not in constants},)

        return var_grid, constants

    def __repr__(self):
        var_grid, constants = self.cv_form()

        return f"GenerationRuns\nvar_grid\n{_indent(pformat(var_grid,))}\nconstants\n{_indent(pformat(constants))})"

    def combine(self, other: 'GenerationRuns') -> 'GenerationRuns':
        param_grid = tuple()

        # not using set to preserve order
        for params in chain(self.param_grid, other.param_grid):
            if params not in param_grid:
                param_grid += (params,)

        return GenerationRuns(param_grid)


class RunDirectory:
    def __init__(self, path: str):
        self.path = path

        self.complete_runs = set()
        self.meta = {}

        self.params_paths = {}
        self.meta_paths = {}

        self.path_feats = os.path.join(self.path, 'feats')
        self.complete_feats = set()

        self.path_metrics = os.path.join(self.path, 'metrics')
        self.complete_metrics = set()

        self.scan()

    def fullpath(self, path):
        return os.path.join(self.path, path)

    def fullpath_feats(self, path):
        return os.path.join(self.path_feats, path)

    def scan(self):
        os.makedirs(self.path, exist_ok=True)

        for fp in os.listdir(self.path):
            full_path = self.fullpath(fp)

            if full_path.endswith('_params.json'):
                try:
                    params = GenerationRunParams.from_json_file(full_path)

                    meta_path = full_path[:-len('_params.json')] + '_meta.json'

                    meta = RunMetadata.from_json_file(meta_path)

                    self.complete_runs.add(params)
                    self.params_paths[params] = full_path

                    self.meta[params] = meta
                    self.meta_paths[params] = meta_path
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

        self.scan_feats()
        self.scan_metrics()

    def scan_feats(self):
        os.makedirs(self.path_feats, exist_ok=True)

        uids_to_params = {params.uid: params for params in self.complete_runs}

        for fp in os.listdir(self.path_feats):
            uid = fp.split("_")[0]
            if uid in uids_to_params:
                self.complete_feats.add(uids_to_params[uid])

    def scan_metrics(self):
        os.makedirs(self.path_metrics, exist_ok=True)

        uids_to_params = {params.uid: params for params in self.complete_runs}

        for fp in os.listdir(self.path_metrics):
            uid = fp.split("_")[0]
            if uid in uids_to_params:
                self.complete_metrics.add(uids_to_params[uid])

    def tokens_path(self, params):
        return self.fullpath(params.uid + '.pt')

    def params_path(self, params):
        return self.fullpath(params.uid + '_params.json')

    def meta_path(self, params):
        return self.fullpath(params.uid + '_meta.json')

    def feats_path(self, params):
        return self.fullpath_feats(params.uid + '_feat.pt')

    def ground_truth_feats_path(self, params):
        return self.fullpath_feats(params.prompt_source_file + '_feat.pt')

    def record(self, params, meta, writefile=True):
        self.complete_runs.add(params)

        if meta is not None:
            self.meta[params] = meta

        if writefile:
            params.to_json_file(self.params_path(params))

            if meta is not None:
                meta.to_json_file(self.meta_path(params))

    def remove(self, params, deletefiles=True):
        if deletefiles:
            os.remove(self.params_paths[params])
            os.remove(self.meta_paths[params])

        self.complete_runs.remove(params)
        del self.meta[params]
        del self.params_paths[params]
        del self.meta_paths[params]

    def record_feats(self, params):
        self.complete_feats.add(params)

    def save_tokens(self, params, tokens):
        import torch as th
        th.save(tokens, self.tokens_path(params))

    def load_tokens(self, params):
        import torch as th
        return th.load(self.tokens_path(params))

    def save_feats(self, params, feats):
        import torch as th
        th.save(feats, self.feats_path(params))

    def save_groundtruth_feats(self, params, feats):
        import torch as th
        th.save(feats, self.ground_truth_feats_path(params))

    def load_feats(self, params):
        import torch as th
        return th.load(self.feats_path(params))
