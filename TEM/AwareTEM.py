import math
import numpy as np
import torch
from gensim.models import KeyedVectors
from pathlib import Path
from typing import Optional, cast

BASE = Path(__file__).resolve().parents[1]  # one level up from TEM/
DATA = BASE / "data"

VOCAB_PATH = DATA / "vocab.txt"
with VOCAB_PATH.open("r", encoding="utf-8") as f:
    VOCAB = {line.strip() for line in f}

EMBED = DATA / "glove.840B.300d_filtered.txt"


class AwareTEM:
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

        self.excellent_in_yball_count = 0
        self.total_words_count = 0
        self.words_changed_to_excellent = []

    def replace_word(self, input_word: str,
                     label: str,
                     trigger_set: set,
                     delta: float = 0.6,
                     lambda_pos: float = 1.0,
                     lambda_neg: float = 1.0) -> str:
        """
        Whitelist-Aware TEM 2:
        - Uses Euclidean base scoring
        - Applies label-aware cosine bias to trigger words
        - Performs exponential mechanism with Gumbel noise
        """

        if input_word not in self.embedding_matrix:
            return input_word  # OOV passthrough

        idx = cast(int, self.embedding_matrix.key_to_index[input_word])  # type: ignore
        vec = self.mat_raw[idx]  # raw vector (Euclidean)
        vec_norm = self.mat_norm[idx]  # normalized vector (cosine)

        dists = torch.norm(self.mat_raw - vec, dim=1)  # (V,)
        cos_sims = torch.matmul(self.mat_norm, vec_norm)  # (V,)

        # Step 1: Adjust scores based on label-aware cosine bias
        # adjusted_scores = -dists.clone()  # base score = -Euclidean

        # for i in range(self.vocab_size):
        #     word = self.words[i]
        #     if word in trigger_set and cos_sims[i] >= delta:
        #         if label == "positive":
        #             # adjusted_scores[i] += lambda_pos * cos_sims[i]
        #             adjusted_scores[i] += lambda_pos
        #         elif label == "negative":
        #             # adjusted_scores[i] -= lambda_neg * cos_sims[i]
        #             adjusted_scores[i] -= lambda_neg
        base_scores = -dists.clone()

        # Step 2: Compute threshold γ for candidate set
        beta = 0.001
        gamma = round((2.0 / self.epsilon) * math.log(((1 - beta) * (self.embed_size - 1)) / beta), 1)

        # Step 3: Build candidate set Lw
        mask = dists <= gamma
        Lw_idxs = mask.nonzero(as_tuple=False).squeeze(1)

        if Lw_idxs.numel() == 0:
            Lw_idxs = torch.arange(self.vocab_size, device=self.device)

        f_Lw = base_scores[Lw_idxs]

        # ✅ STEP B: Apply bias if in trigger_set and valid
        if "excellent" in trigger_set:
            excellent_idx = self.embedding_matrix.key_to_index["excellent"]  # type: ignore
            matches = (Lw_idxs == excellent_idx)
            cos_mask = cos_sims[excellent_idx] >= delta
            if matches.any() and cos_mask:
                idx_in_f_Lw = matches.nonzero(as_tuple=True)[0][0]
                if label == "pos":
                    f_Lw[idx_in_f_Lw] += lambda_pos
                elif label == "neg":
                    f_Lw[idx_in_f_Lw] -= lambda_neg

        # Step 4: Add ⊥ score
        num_outside = self.vocab_size - Lw_idxs.numel()
        if num_outside > 0:
            f_bottom = -gamma + (2.0 / self.epsilon) * math.log(num_outside)
        else:
            f_bottom = float('-inf')  # no outside candidates

        scores = torch.cat([f_Lw, torch.tensor([f_bottom], device=self.device)], dim=0)

        # Step 5: Add Gumbel noise and sample
        gumbel = torch.distributions.Gumbel(0.0, 2.0 / self.epsilon)
        raw_noise = gumbel.sample(sample_shape=scores.shape)  # type: ignore
        noise = cast(torch.Tensor, raw_noise).to(self.device)
        noisy_scores = scores + noise

        choice = int(noisy_scores.argmax().item())
        if choice == f_Lw.size(0):
            # ⊥ chosen → sample from outside
            outside = (~mask).nonzero(as_tuple=False).squeeze(1)
            r = int(torch.randint(0, outside.numel(), (1,), device=self.device).item())
            chosen_idx = int(outside[r].item())
        else:
            chosen_idx = int(Lw_idxs[choice].item())

        chosen_word = self.words[chosen_idx]

        # Track all replacements to 'excellent' (including identity)
        if input_word == "great":
            print(f"  Final selected word: {chosen_word}")

        return cast(str, self.words[chosen_idx])

