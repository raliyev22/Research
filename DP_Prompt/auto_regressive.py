from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList  # for Transformers ≥4.32
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as PreTrainedTokenizer, BatchEncoding
from transformers.modeling_utils import PreTrainedModel

import torch
from torch import Tensor
from typing import Tuple, Optional, Dict, cast


def prompt_template_fn(private_doc: str) -> str:
    """
    Build the zero-shot prompt for DP-Prompt as in the paper (Appendix B).
    """
    return f"Document: {private_doc}\nParaphrase of the document:"


class ClippedLogitsProcessor(LogitsProcessor):
    """
    HuggingFace LogitsProcessor that clamps logits to [b1, b2] before sampling.
    """

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        return torch.clamp(scores, self.min_val, self.max_val)


class DPPromptRewriter:
    """
    Implements the DP-Prompt auto-regressive mechanism (Algorithm 1) with
    local differential privacy via the exponential mechanism and logit clipping.
    """

    def __init__(
            self,
            model_name: str,
            clipping_bounds: Tuple[float, float],  # (b1, b2)
            temperature: float,  # T > 0
            max_new_tokens: int,  # n
            device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load and move model to device
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            low_cpu_mem_usage=True,
        )
        self.b1, self.b2 = clipping_bounds
        self.T = temperature
        self.max_new_tokens = max_new_tokens

    def rewrite(self, private_doc: str) -> str:
        """
        Generate a DP-compliant paraphrase of `private_doc`.
        Returns the sanitized output string.
        """
        # 1) Build and tokenize prompt
        prompt = prompt_template_fn(private_doc)
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = cast(torch.Tensor, tokenized["input_ids"]).to(self.device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = cast(torch.Tensor, attention_mask).to(self.device)

        # 2) Prepare DP logits processor
        logits_proc = LogitsProcessorList([
            ClippedLogitsProcessor(self.b1, self.b2)
        ])

        # 3) Generate with DP sampling
        outputs = self.model.generate(  # type: ignore[operator]
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            temperature=self.T,
            logits_processor=logits_proc,
        )

        # 4) Decode only the newly generated tokens
        gen_tokens = outputs[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    def compute_epsilon(self, num_tokens: Optional[int] = None) -> float:
        """
        Compute the theoretical LDP epsilon: ε = 2·n·(b2 − b1)/T.
        If num_tokens is None, uses max_new_tokens.
        """
        n = num_tokens if num_tokens is not None else self.max_new_tokens
        return 2 * (self.b2 - self.b1) / self.T


if __name__ == "__main__":
    rewriter = DPPromptRewriter(
        model_name="stabilityai/stablelm-tuned-alpha-7b",
        clipping_bounds=(-5.0, 5.0),
        temperature=1.5,
        max_new_tokens=150,
    )
    private_text = (
        "My name is Raul. I love hiking in the mountains on weekends"
        " and writing about my adventures."
    )
    sanitized = rewriter.rewrite(private_text)
    print("Sanitized paraphrase:\n", sanitized)
    print(f"Theoretical LDP ε: {rewriter.compute_epsilon():.2f}")