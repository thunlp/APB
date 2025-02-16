
from typing import List


def select_tokenizer(tokenizer_type, tokenizer_path):
    if tokenizer_type == 'hf':
        return HFTokenizer(model_path=tokenizer_path)
    else:
        raise ValueError(f'Unknown tokenizer_type {tokenizer_type}')


class HFTokenizer:
    """
    Tokenizer from HF models
    """

    def __init__(self, model_path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text
