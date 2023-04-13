from typing import List, Union, Optional

import numpy as np

from transformers import BatchEncoding
from transformers import AutoTokenizer, PreTrainedTokenizer


def timestampify(
    start: np.ndarray,
    end: np.ndarray,
    duration: np.ndarray,
    num_bins: int = 100,
    vocabulary_size: int = 0,
) -> np.ndarray:
    """
    Convert start and end timestamps to tokens.
    Args:
        start (np.ndarray): Start times of the events. [t, 1]
        end (np.ndarray): End times of the events. [t, 1]
        duration (np.ndarray): Duration of the video. [1]
        num_bins (int): Number of quantization bins for time tokens. Default: 100.
        vocabulary_size (int): Number of text tokens. Default: 0.
    Returns:
        timestamp_token (np.ndarray): Time tokens. [t, 2]
    """

    timestamp = np.stack([start, end], axis=1)
    timestamp = np.minimum(timestamp, duration)

    max_offset = np.float64(num_bins - 1)
    rel_timestamp = np.divide(timestamp, duration)
    timestamp_token = np.add(np.multiply(rel_timestamp, max_offset), vocabulary_size)
    timestamp_token = timestamp_token.astype(np.int64)

    return timestamp_token


def merge_tokens(
    caption_tokens: List[List[int]],
    timestamp_token: np.ndarray,
) -> np.ndarray:
    """
    Merge text and time tokens.
    Args:
        caption_tokens (list): List of text tokens.
        timestamp_token (np.ndarray): Time tokens of the events.
    Returns:
        merged_tokens (np.ndarray): Merged tokens.
    """
    merged_tokens = []
    for i, tokens in enumerate(caption_tokens):
        merged_tokens.extend(timestamp_token[i])
        merged_tokens.extend(tokens)
    merged_tokens = np.array(merged_tokens)
    return merged_tokens


class Tokenizer:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, str],
        num_bins: int = 100,
    ):
        """
        Args:
            tokenizer (PreTrainedTokenizerBase or str): Tokenizer.
            num_bins (int): Number of quantization bins for time tokens. Default: 100.
        """
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.num_bins = num_bins
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

    def __call__(
        self,
        text: Union[str, List[str]],
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
        duration: Optional[np.ndarray] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ):
        if isinstance(text, str):
            text = [text]

        caption_tokens = self.tokenizer(text)["input_ids"]

        if start is None or end is None or duration is None:
            return caption_tokens

        timestamp_token = timestampify(
            start, end, duration, self.num_bins, self.vocab_size
        )
        merged_tokens = merge_tokens(caption_tokens, timestamp_token)

        if truncation and max_length is not None:
            merged_tokens = merged_tokens[:max_length]

        batch = {
            "input_ids": merged_tokens,
            "attention_mask": [1] * len(merged_tokens),
        }

        if padding and max_length is not None:
            batch["input_ids"] = np.concatenate(
                [
                    batch["input_ids"],
                    [self.pad_token_id] * (max_length - len(batch["input_ids"])),
                ]
            )
            batch["attention_mask"] = np.concatenate(
                [
                    batch["attention_mask"],
                    [0] * (max_length - len(batch["attention_mask"])),
                ]
            )

        return BatchEncoding(batch, tensor_type=return_tensors)
