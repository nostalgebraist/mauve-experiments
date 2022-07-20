import math

import torch
from transformers import LogitsProcessor, LogitsWarper


class BreakrunsLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 base_temperature: float,
                 tau: float,
                 debug=True,
                 tokenizer=None
                 ):
        self.base_temperature = base_temperature
        self.tau = tau
        self.debug = debug
        self.tokenizer = tokenizer

        self.breakruns_counter = None
        self.last_logits = None
        self.last_length = None

    def _reset(self):
        self._dprint("BREAKRUNS: _reset")
        self.breakruns_counter = None
        self.last_logits = None

    def _dprint(self, msg, fillers={}, **kwargs):
        if self.debug:
            print(msg.format(**fillers), **kwargs)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_length = input_ids.shape[1]
        if seq_length < 1:
            self._dprint("BREAKRUNS: empty sequence, no op")
            return scores

        if self.last_length is None or self.last_length > input_ids.shape[1]:
            # new sequence
            self._reset()
        self.last_length = input_ids.shape[1]

        if self.breakruns_counter is None:
            self._dprint("BREAKRUNS: init counter")
            self.breakruns_counter = torch.zeros((), device=input_ids.device)

        if self.last_logits is None:
            self._dprint("BREAKRUNS: init logits, no op")
            self.last_logits = scores

            return scores

        # check if last was top
        was_top = (input_ids[:, -1] == self.last_logits.argmax(dim=1)).to(torch.long)

        self.breakruns_counter = was_top * (self.breakruns_counter + 1)

        if self.debug:
            sampled_str = repr(self.tokenizer.decode(input_ids[0, -1].item()))
            actual_top_str = repr(self.tokenizer.decode([self.last_logits.argmax(dim=1)[0].item()]))
            print(f"was_top?: {was_top[0]} | sampled {sampled_str} actual_top {actual_top_str} | self.breakruns_counter: {self.breakruns_counter}")

        eff_temperature = self.base_temperature + (self.breakruns_counter * self.tau)
        self._dprint("eff_temperature: {et}", fillers={"et": eff_temperature})

        self.last_logits = scores

        return scores / eff_temperature[:, None].expand_as(scores)


class MirostatLogitsProcessor(LogitsProcessor):
    def __init__(self, tau, n=50000, learning_rate=1):
        self.tau = tau
        self.max_surprise = 2*self.tau
        self.n=50000
        self.learning_rate = learning_rate

        self.last_length = None

    def _reset(self):
        self.max_surprise = 2*self.tau

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        logits = scores
        context = input_ids

        if self.last_length is None or self.last_length > input_ids.shape[1]:
            # new sequence
            self._reset()
        self.last_length = input_ids.shape[1]

        sorted_logits, _ = torch.sort(logits, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()

        # Estimate s
        s = [self.estimate_s(p) for p in prob_original]
        # Compute k
        k = [self.compute_k(se,ms)+1 for se, ms in zip(s, self.max_surprise)]

        sorted_logits = torch.cat([sl[0:kk] for sl, kk in zip(sorted_logits, k)])

        prob_topk = torch.softmax(sorted_logits, dim = 0)
        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
        index_surprise = math.log2(1/prob_original[prev_i])

        # adjust max_surprise
        self.error_surprise = index_surprise - self.tau
        self.max_surprise -= self.learning_rate*error_surprise
        self.max_surprise = self.max_surprise - error_surprise

    @staticmethod
    def estimate_s(prob):
        result = 0
        num = 0
        den = 0
        for i in range(100):
            b = prob[i]/prob[i+1]
            t = (i+2)/(i+1)
            num += math.log(b)*math.log(t)
            den += math.log(t)**2
        return num/den

    def compute_k(s):
        eps = s-1
        k = ((eps*(2**(self.tau)))/(1-self.n**(-eps)))**(1/s)
        k = round(k)
        return k
