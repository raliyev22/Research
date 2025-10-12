# whitelist_aware_tem1.py
import math
import torch
import numpy as np
from gensim.models import KeyedVectors
from pathlib import Path
from typing import Optional, cast

class WhitelistAwareTEM1:
    def __init__(self,
                 epsilon: float,
                 whitelist_path: Path,
                 embed_path: Path,
                 device: Optional[str] = None):
        self.epsilon = epsilon
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load GloVe vectors
        self.embedding_matrix = KeyedVectors.load_word2vec_format(str(embed_path), binary=False, unicode_errors="ignore")
        self.words = self.embedding_matrix.index_to_key
        self.vocab_size = len(self.words)

        raw = torch.from_numpy(self.embedding_matrix.vectors.astype(np.float32)).to(self.device)
        self.mat_raw = raw  # Euclidean distances
        self.embed_size = self.mat_raw.size(0)

        # Load whitelist from .txt
        with open(whitelist_path, "r", encoding="utf-8") as f:
            self.whitelist = set(w.strip() for w in f if w.strip())
        self.whitelist_indices = {i for i, w in enumerate(self.words) if w in self.whitelist}

    def replace_word(self, input_word: str) -> str:
        if input_word not in self.embedding_matrix:
            return input_word

        idx = cast(int, self.embedding_matrix.key_to_index[input_word])
        vec = self.mat_raw[idx]
        dists = torch.norm(self.mat_raw - vec, dim=1)

        # Compute γ
        beta = 0.001
        gamma = round((2.0 / self.epsilon) * math.log(((1 - beta) * (self.embed_size - 1)) / beta), 1)

        # Determine candidate set
        if input_word in self.whitelist:
            candidate_indices = [i for i in self.whitelist_indices if dists[i] <= gamma]
        else:
            candidate_indices = (dists <= gamma).nonzero(as_tuple=False).squeeze(1).tolist()

        if not candidate_indices:
            candidate_indices = list(range(self.vocab_size))
        f_Lw = -dists[candidate_indices]

        # ⊥ (null) scoring
        num_outside = self.vocab_size - len(candidate_indices)
        f_bottom = -gamma + (2.0 / self.epsilon) * math.log(num_outside) if num_outside > 0 else float('-inf')

        # Add Gumbel noise and sample
        scores = torch.cat([f_Lw, torch.tensor([f_bottom], device=self.device)], dim=0)
        gumbel = torch.distributions.Gumbel(0.0, 2.0 / self.epsilon)
        noise = torch.tensor(gumbel.sample(sample_shape=scores.shape), device=self.device)
        noisy_scores = scores + noise

        choice = int(noisy_scores.argmax().item())
        if choice == len(candidate_indices):  # ⊥ chosen
            outside_indices = [i for i in range(self.vocab_size) if i not in candidate_indices]
            chosen_idx = int(np.random.choice(outside_indices))
        else:
            chosen_idx = candidate_indices[choice]

        return self.words[chosen_idx]
