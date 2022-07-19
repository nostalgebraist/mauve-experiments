import torch as th
from tqdm.auto import tqdm, trange

from src.nost.generation_config import GenerationRunParams, GenerationRuns, RunMetadata, RunDirectory
from src.nost.compute_mauve_from_package import get_features_from_input
from src.nost.util import load_ground_truth

# load_ground_truth(enc, data_dir, prompt_source_file, min_len

def featurize_tokens(tokens, max_examples, batch_size, featurize_model_name='gpt2-large', device_id=0):
    return get_features_from_input(
        features=None,
        tokenized_texts=tokens[:max_examples],
        texts=None,
        featurize_model_name=featurize_model_name,
        maxlen=None,  # unused
        name='',
        batch_size=batch_size,
        verbose=True,
        device_id=0
    )


class Featurizer:
    def __init__(
        self, run_directory: Union[str, RunDirectory], runs: GenerationRuns, device_id=0, data_dir='data',
        featurize_model_name='gpt2-large',
    ):
        self.runs = runs
        self.device_id = device_id
        self.data_dir = data_dir
        self.featurize_model_name = featurize_model_name

        if isinstance(run_directory, str):
            run_directory = RunDirectory(path=run_directory)

        self.run_directory = run_directory

    @property
    def complete_feats(self):
        return self.run_directory.complete_feats.intersection(self.runs.param_grid)

    def feats_to_do(self):
        return [r for r in self.runs.param_grid if r not in self.complete_feats]

    def featurize_run(self, params: GenerationRunParams, batch_size: int):
        self.featurize_ground_truth(params, batch_size)  # skips internally if already done

        tokens = self.run_directory.load_tokens(params)

        feat = featurize_tokens(
            tokens,
            max_examples=params.max_num_generations,
            batch_size=batch_size,
            featurize_model_name=self.featurize_model_name,
            device_id=self.device_id,
        )

        self.run_directory.save_feats(params, feats)
        self.run_directory.record_feats(params)


    def featurize_ground_truth(self, params: GenerationRunParams, batch_size: int):
        if os.path.exists(self.run_directory.ground_truth_feats_path(params)):
            return

        tokens = load_ground_truth(
            enc=None,  # should be pre-tokenized
            data_dir=self.data_dir,
            prompt_source_file=params.prompt_source_file,
            min_len=params.prompt_len
        )

        tokens = [seq[:, :params.max_len] for seq in tokens]

        feat = featurize_tokens(
            tokens,
            max_examples=params.max_num_generations,
            batch_size=batch_size,
            featurize_model_name=self.featurize_model_name,
            device_id=self.device_id,
        )

        self.run_directory.save_groundtruth_feats(params, feats)
