"""
Frozen Pretrained Text Feature Extractor for Multimodal Knowledge Tracing.

Extracts both token-level hidden states (for ACD-MMKTv3 attention pooling and
cross-modal alignment) and masked mean-pooled global vectors (for MG-MMKT/v2),
while providing exact subword token strings aligned 1-to-1 with token masks for
Tier 3 explainability.
"""

from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModel, AutoTokenizer


class FrozenTextEncoder:
    """
    Frozen Hugging Face Transformer encoder producing token-level and pooled representations.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        device: Optional[torch.device] = None,
        local_files_only: bool = True,
    ):
        """
        Initialize frozen text backbone and tokenizer.

        Args:
            model_name: Pretrained Hugging Face model name or local path.
            device: Target compute device; auto-selected if None.
            local_files_only: Prefer local HF cache before falling back to remote download.
        """
        self.model_name = model_name
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            self.model = AutoModel.from_pretrained(
                model_name, local_files_only=local_files_only
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        self.model = self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_texts(
        self,
        texts: List[str],
        max_length: int = 128,
        batch_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[str]]]:
        """
        Encode a list of texts into token states, validity masks, pooled vectors, and token lists.

        Args:
            texts: List of input text strings of length N.
            max_length: Padded/truncated token sequence length L.
            batch_size: Inference mini-batch size.

        Returns:
            Tuple of:
            - token_states: [N, L, d_txt] float16 tensor.
            - token_masks: [N, L] bool tensor (True for real tokens, False for padding).
            - pooled_vectors: [N, d_txt] float16 masked mean-pooled global representations.
            - token_strings: List of N lists, each containing L aligned subword strings.
        """
        num_items = len(texts)
        hidden_dim = int(self.model.config.hidden_size)

        token_states = torch.zeros((num_items, max_length, hidden_dim), dtype=torch.float16)
        token_masks = torch.zeros((num_items, max_length), dtype=torch.bool)
        pooled_vectors = torch.zeros((num_items, hidden_dim), dtype=torch.float16)
        token_strings: List[List[str]] = []

        with torch.no_grad():
            for start in range(0, num_items, batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids_cpu = encoded["input_ids"]
                for row_ids in input_ids_cpu.tolist():
                    token_strings.append(self.tokenizer.convert_ids_to_tokens(row_ids))

                encoded_dev = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded_dev)
                hidden = outputs.last_hidden_state
                mask = encoded_dev["attention_mask"].bool()

                # Masked mean pooling strictly over non-padding tokens
                mask_f = mask.unsqueeze(-1).float()
                sum_emb = (hidden.float() * mask_f).sum(dim=1)
                denom = mask_f.sum(dim=1).clamp(min=1e-8)
                pooled = sum_emb / denom

                end = start + len(batch_texts)
                token_states[start:end] = hidden.cpu().to(torch.float16)
                token_masks[start:end] = mask.cpu()
                pooled_vectors[start:end] = pooled.cpu().to(torch.float16)

        return token_states, token_masks, pooled_vectors, token_strings

    def tokenize_only(self, texts: List[str], max_length: int = 128) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        Run deterministic tokenization without neural forward pass to verify 1-to-1 alignment.

        Returns:
            Tuple of (attention_mask [N, L] bool tensor, token_strings list).
        """
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        attn_mask = encoded["attention_mask"].bool()
        token_strings = [
            self.tokenizer.convert_ids_to_tokens(row)
            for row in encoded["input_ids"].tolist()
        ]
        return attn_mask, token_strings

    def release(self) -> None:
        """Free model weights from device memory after offline extraction."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
