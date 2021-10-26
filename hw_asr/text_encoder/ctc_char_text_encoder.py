from typing import List, Tuple

import torch
from ctcdecode import CTCBeamDecoder

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from hw_asr.utils.util import init_lm


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        init_lm()
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.beam_search = CTCBeamDecoder(
            ['^'] + alphabet,
            model_path='lowercase_3-gram.pruned.1e-7.arpa',
            alpha=0.4,
            beta=1.0,
            beam_width=100,
            log_probs_input=True
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

    def ctc_beam_search(self, log_probs) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        # assert len(log_probs.shape) == 2
        # char_length, voc_size = log_probs.shape
        # assert voc_size == len(self.ind2char)

        beam_results, beam_scores, timesteps, out_lens = self.beam_search.decode(log_probs)
        # print(beam_results.size())
        res = ''.join([self.ind2char[int(i)] for i in beam_results[0][0][:out_lens[0][0]]])
        # print(''.join([self.ind2char[int(i)] for i in beam_results[0][0][:out_lens[0][0]]]))
        return res

    def ctc_beam_search_test(self, log_probs) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        # assert len(log_probs.shape) == 2
        # char_length, voc_size = log_probs.shape
        # assert voc_size == len(self.ind2char)

        beam_results, beam_scores, timesteps, out_lens = self.beam_search.decode(log_probs)
        # print(beam_results.size())
        res = []
        for j in range(100):
            res.append(''.join([self.ind2char[int(i)] for i in beam_results[0][j][:out_lens[0][j]]]))
        # print(''.join([self.ind2char[int(i)] for i in beam_results[0][0][:out_lens[0][0]]]))
        return res
