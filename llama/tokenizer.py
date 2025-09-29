# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    Collection,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
    cast,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer

logger = getLogger(__name__)

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: str
    content: str

Dialog = Sequence[Message]


class Tokenizer:
    """Unified tokenizer supporting Llama (tiktoken) and Qwen (HuggingFace)."""

    def __init__(self, model_path: str, model_type: str = "llama"):
        """
        Initialize tokenizer.

        Args:
            model_path: For Llama: path to .model file. For Qwen: path to model directory.
            model_type: "llama" or "qwen"
        """
        assert model_type in ("llama", "qwen"), f"Unsupported model type: {model_type}"

        self.model_type = model_type

        if model_type == "qwen":
            model_dir = Path(model_path)
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_path}")

            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.n_words = self.hf_tokenizer.vocab_size
            self.bos_id = getattr(self.hf_tokenizer, 'bos_token_id', 0) or 0
            self.eos_id = getattr(self.hf_tokenizer, 'eos_token_id', 0) or 0
            self.pad_id = getattr(self.hf_tokenizer, 'pad_token_id', -1) or -1

            self.special_tokens = {}
            for token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
                if token in self.hf_tokenizer.get_vocab():
                    self.special_tokens[token] = self.hf_tokenizer.convert_tokens_to_ids(token)

            self.eot_id = self.special_tokens.get("<|im_end|>", self.eos_id)
            self.stop_tokens = {self.eos_id, self.eot_id}
            if "<|endoftext|>" in self.special_tokens:
                self.stop_tokens.add(self.special_tokens["<|endoftext|>"])

        else:
            assert os.path.isfile(model_path), model_path

            mergeable_ranks = load_tiktoken_bpe(model_path)
            num_base_tokens = len(mergeable_ranks)

            special_tokens = [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|reserved_special_token_2|>",
                "<|reserved_special_token_3|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|reserved_special_token_4|>",
                "<|eot_id|>",
            ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 251)]

            self.special_tokens = {
                token: num_base_tokens + i for i, token in enumerate(special_tokens)
            }

            self.tiktoken_model = tiktoken.Encoding(
                name=Path(model_path).name,
                pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                mergeable_ranks=mergeable_ranks,
                special_tokens=self.special_tokens,
            )

            self.n_words = self.tiktoken_model.n_vocab
            self.bos_id = self.special_tokens["<|begin_of_text|>"]
            self.eos_id = self.special_tokens["<|end_of_text|>"]
            self.eot_id = self.special_tokens["<|eot_id|>"]
            self.pad_id = -1
            self.stop_tokens = {self.eos_id, self.eot_id}

        logger.info(f"Loaded {model_type} tokenizer from {model_path}")

    def encode(
        self,
        s: str,
        *,
        bos: bool = False,
        eos: bool = False,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        if self.model_type == "qwen":
            tokens = self.hf_tokenizer.encode(s, add_special_tokens=False)

            if bos and self.bos_id is not None:
                tokens = [self.bos_id] + tokens
            if eos and self.eos_id is not None:
                tokens = tokens + [self.eos_id]

            return tokens
        else:
            TIKTOKEN_MAX_ENCODE_CHARS = 400_000
            MAX_NO_WHITESPACES_CHARS = 25_000

            substrs = (
                substr
                for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
                for substr in self._split_whitespaces_or_nonwhitespaces(
                    s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
                )
            )

            tokens = []
            for substr in substrs:
                tokens.extend(
                    self.tiktoken_model.encode(
                        substr,
                        allowed_special=allowed_special,
                        disallowed_special=disallowed_special,
                    )
                )

            if bos:
                tokens.insert(0, self.bos_id)
            if eos:
                tokens.append(self.eos_id)

            return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        if self.model_type == "qwen":
            return self.hf_tokenizer.decode(list(tokens), skip_special_tokens=False)
        else:
            return self.tiktoken_model.decode(cast(List[int], tokens))

    def apply_chat_template(self, messages: List[Message], **kwargs) -> List[int]:
        if self.model_type != "qwen":
            raise NotImplementedError("Chat templates only supported for Qwen models")

        return self.hf_tokenizer.apply_chat_template(messages, return_tensors=None, **kwargs)

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


class QwenChatFormat:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        assert tokenizer.model_type == "qwen", "QwenChatFormat requires Qwen tokenizer"

    def encode_header(self, message: Message) -> List[int]:
        header_text = f"<|im_start|>{message['role']}\n"
        return self.tokenizer.encode(header_text, bos=False, eos=False)

    def encode_message(self, message: Message) -> List[int]:
        return self.tokenizer.apply_chat_template([message], add_generation_prompt=False, enable_thinking=False)

    def encode_dialog(self, dialog: Dialog) -> List[int]:
        return self.tokenizer.apply_chat_template(list(dialog), add_generation_prompt=False, enable_thinking=False)

    def encode_dialog_prompt(self, dialog: Dialog, prefill: bool = True) -> List[int]:
        return self.tokenizer.apply_chat_template(list(dialog), add_generation_prompt=prefill, enable_thinking=False)


class LlamaChatFormat:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        assert tokenizer.model_type == "llama", "LlamaChatFormat requires Llama tokenizer"

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(self.tokenizer.encode(message["content"].strip(), bos=False, eos=False))
        if tokens[-1] != self.tokenizer.special_tokens["<|eot_id|>"]:
            tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialog(self, dialog: Dialog) -> List[int]:
        tokens = []
        for message in dialog:
            tokens.extend(self.encode_message(message))
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog, prefill: bool = True) -> List[int]:
        tokens = [self.tokenizer.special_tokens["<|begin_of_text|>"]]
        tokens.extend(self.encode_dialog(dialog))
        if prefill:
            tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


class ChatFormat:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        if tokenizer.model_type == "qwen":
            self._formatter = QwenChatFormat(tokenizer)
        else:
            self._formatter = LlamaChatFormat(tokenizer)

    def encode_header(self, message: Message) -> List[int]:
        return self._formatter.encode_header(message)

    def encode_message(self, message: Message) -> List[int]:
        return self._formatter.encode_message(message)

    def encode_dialog(self, dialog: Dialog) -> List[int]:
        return self._formatter.encode_dialog(dialog)

    def encode_dialog_prompt(self, dialog: Dialog, prefill: bool = True) -> List[int]:
        return self._formatter.encode_dialog_prompt(dialog, prefill)
