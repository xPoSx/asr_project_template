from typing import List, Tuple

import torch
import pyctcdecoder

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.beam_search = pyctcdecoder.build_ctcdecoder(
            alphabet,
            'lowercase_3-gram.pruned.1e-7.arpa'
        )

    def ctc_decode(self, inds: List[int]) -> str:
        res = ""
        for i, ind in enumerate(inds):
            ind = int(ind)
            if i > 0 and ind == inds[i - 1] or ind == 0:
                continue
            else:
                res += self.ind2char[ind]
        return res

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        print(self.beam_search.forward(log_probs = np.expand_dims(probs, axis=0), log_probs_length=probs_length))
        raise RuntimeError('kek')
        return sorted(hypos, key=lambda x: x[1], reverse=True)
