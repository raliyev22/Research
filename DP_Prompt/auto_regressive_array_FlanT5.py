from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torch import Tensor
from typing import Optional
import numpy as np

def prompt_template_fn(private_doc: str) -> str:
    return f"Document: {private_doc}\nParaphrase of the document:"

class VectorClippedLogitsProcessor(LogitsProcessor):
    def __init__(self, min_vals: Tensor, max_vals: Tensor):
        self.min_vals = min_vals.view(1, -1)  # (1, vocab_size)
        self.max_vals = max_vals.view(1, -1)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        # scores: (batch_size, vocab_size)
        return torch.clamp(scores, min=self.min_vals, max=self.max_vals)


class DPPromptRewriter:
    def __init__(
        self,
        model_name: str,
        min_vals_path: str,
        max_vals_path: str,
        temperature: float,
        max_new_tokens: int,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            low_cpu_mem_usage=True,
        ).to(self.device)
        # ensure pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 2) Load per-token clip bounds
        min_vals = torch.from_numpy(np.load(min_vals_path)).to(self.device)
        max_vals = torch.from_numpy(np.load(max_vals_path)).to(self.device)
        self.logits_processor = LogitsProcessorList([
            VectorClippedLogitsProcessor(min_vals, max_vals)
        ])

        self.T = temperature
        self.max_new_tokens = max_new_tokens

    def rewrite(self, private_doc: str) -> str:
        prompt = prompt_template_fn(private_doc)
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask", None),
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.T,
            top_k=0,
            top_p=1.0,
            logits_processor=self.logits_processor,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def compute_epsilon(self) -> float:
        # if you need the theoretical ε, you can still use some scalar bound—for the vector case you'd
        # take max(max_vals-min_vals) as your worst-case Δ, but usually you report per-dimension.
        raise NotImplementedError("Vector case: compute ε per token or use max range manually.")

if __name__ == "__main__":
    rewriter = DPPromptRewriter(
        model_name = "google/flan-t5-xl",
        min_vals_path="cleaned_yelp_min_logits_FlanT5.npy",
        max_vals_path="cleaned_yelp_max_logits_FlanT5.npy",
        temperature=1.25,
        max_new_tokens=50,
    )

    private_text = (
        "So upset I do not know where to begin. I will call the store later today and try to get my issues resolved, but basically they wanted to do $1300 - $1400 worth of (according to my dealer) un-needed repairs and they did a shoddy job on my oil change. I would up paying for a 2nd oil change and a completely different roster of repairs at the dealership. I am very hesitant to ever come back here."
    )
    # private_text = (
    #     "My name is Raul and I cannot solve this problem."
    # )
    paraphrase = rewriter.rewrite(private_text)
    print("Sanitized paraphrase:\n", paraphrase)
