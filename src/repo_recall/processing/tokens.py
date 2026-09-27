"""Token counting, behind a small interface so tests don't need a real tokenizer."""

from typing import Protocol


class TokenCounter(Protocol):
    """Anything that can say how many tokens a piece of text costs."""

    def count(self, text: str) -> int:
        """Return the number of tokens in `text` (excluding special tokens)."""
        ...


class HFTokenCounter:
    """Counts tokens with the embedding model's own Hugging Face tokenizer.

    Using the model's tokenizer (not a word-count estimate) is what makes a
    chunk-size limit a guarantee that nothing gets truncated at embed time.
    The tokenizer is loaded on first use.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._tokenizer = None

    def count(self, text: str) -> int:
        """Return the number of model tokens in `text`, without [CLS]/[SEP]."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        # verbose=False: counting over-long text is expected here (it's how we
        # decide to split it), so the "longer than max length" warning is noise.
        return len(self._tokenizer.encode(text, add_special_tokens=False, verbose=False))
