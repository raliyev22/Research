#https://github.com/sjmeis/MLDP/blob/main/MLDP.py
import math
import numpy as np
import torch
from gensim.models import KeyedVectors
from pathlib import Path
from typing import Optional, cast

BASE = Path(__file__).parent
DATA = BASE / "data"

VOCAB_PATH = DATA / "vocab.txt"
with VOCAB_PATH.open("r", encoding="utf-8") as f:
    VOCAB = {line.strip() for line in f}

EMBED = DATA / "glove.840B.300d_filtered.txt"


class TEM:
    def __init__(self,
                 epsilon: float,
                 embedding_matrix: Optional[KeyedVectors] = None,
                 embed_path: Path = EMBED,
                 return_noise: bool = False):
        self.epsilon = epsilon
        self.return_noise = return_noise

        if embedding_matrix is not None:
            self.embedding_matrix = embedding_matrix
        else:
            self.embedding_matrix = KeyedVectors.load_word2vec_format(
                str(embed_path), binary=False, unicode_errors="ignore"
            )
        self.embed_size = len(self.embedding_matrix)  # |𝓦| in the paper
        self.words = self.embedding_matrix.index_to_key
        self.vocab_size = len(self.words)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw = torch.from_numpy(self.embedding_matrix.vectors.astype(np.float32)).to(self.device)
        self.mat_raw = raw  # for Euclidean distances
        norms = raw.norm(dim=1, keepdim=True)
        self.mat_norm = raw / norms  # for cosine similarity

    def replace_word(self, input_word: str) -> str:
        """
        Exponential Mechanism over full vocab using Euclidean metric:
        - Compute γ threshold with β=0.001
        - Build candidate set L_w
        - Score via -distance and ⊥ option
        - Add Gumbel noise (scale=2/ε) and pick argmax
        """
        # OOV passthrough
        if input_word not in self.embedding_matrix:
            return input_word

        # Compute distances
        idx = cast(int, self.embedding_matrix.key_to_index[input_word])  # type: ignore
        vec = self.mat_raw[idx]
        dists = torch.norm(self.mat_raw - vec, dim=1)  # (V,)
        # Threshold γ
        beta = 0.001
        # gamma = round((2.0 / self.epsilon) * math.log(((1 - beta) * (V - 1)) / beta), 1)
        gamma = round((2.0 / self.epsilon) * math.log(((1 - beta) * (self.embed_size - 1)) / beta), 1)

        # Candidate set L_w
        mask = dists <= gamma
        Lw_idxs = mask.nonzero(as_tuple=False).squeeze(1)
        if Lw_idxs.numel() == 0:
            Lw_idxs = torch.arange(self.vocab_size, device=self.device)
        f_Lw = -dists[Lw_idxs]

        # ⊥ score for outside
        num_outside = self.vocab_size - Lw_idxs.numel()
        if num_outside > 0:
            f_bottom = -gamma + (2.0 / self.epsilon) * math.log(num_outside)
        else:
            # no outside candidates → force choosing from Lw
            f_bottom = float('-inf')
        # previous version
        # f_bottom = -gamma + (2.0 / self.epsilon) * math.log((V - Lw_idxs.numel()) / 1.0)

        # Combine scores
        scores = torch.cat([f_Lw, torch.tensor([f_bottom], device=self.device)], dim=0)
        gumbel = torch.distributions.Gumbel(0.0, 2.0 / self.epsilon)
        raw_noise = gumbel.sample(sample_shape=scores.shape)  # type: ignore
        noise = cast(torch.Tensor, raw_noise).to(self.device)
        noisy_scores = scores + noise

        # Choose word
        choice = int(noisy_scores.argmax().item())
        if choice == f_Lw.size(0):
            # ⊥ chosen -> sample outside
            outside = (~mask).nonzero(as_tuple=False).squeeze(1)
            r = int(torch.randint(0, outside.numel(), (1,), device=self.device).item())
            chosen_idx = int(outside[r].item())
        else:
            chosen_idx = int(Lw_idxs[choice].item())

        return cast(str, self.words[chosen_idx])
