import os
from typing import Iterable, Iterator, Self
import regex as re
from more_itertools import peekable

from icecream import ic
ic.configureOutput(includeContext=True)

from cs336_basics._cs336_a1_rust import rust_run_train_bpe, read_vocab, read_merges

__all__ = ["run_train_bpe"]


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return rust_run_train_bpe(input_path, vocab_size, special_tokens)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.special_tokens = special_tokens or []
        self.vocab = vocab
        self.inv_vocab = {value: key for key, value in self.vocab.items()}
        self.merges = merges
        # Add special tokens to vocab if not already contained
        for st in self.special_tokens:
            stb = bytes(st, 'utf-8')
            if stb in self.inv_vocab:
                continue
            else:
                id = len(self.vocab)
                self.vocab[id] = stb
                self.inv_vocab[stb] = id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        instance = cls(read_vocab(vocab_filepath), read_merges(merges_filepath), special_tokens)
        return instance

    def encode(self, text: str) -> list[int]:
        prepretoks = []
        if len(self.special_tokens) > 0:
            # Prevent greedy matching by sorting longer special tokens first
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_tok_pat = '(' + '|'.join([re.escape(tok) for tok in sorted_special]) + ')'
            prepretoks = re.split(special_tok_pat, text)
        else:
            prepretoks = [text]

        pretok_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        toks = []
        for prepretok in prepretoks:
            if prepretok in self.special_tokens:
                prepretok_bytes = bytes(prepretok, 'utf-8')
                idx = self.inv_vocab[prepretok_bytes]
                toks.append(idx)
            else:
                for pretok in re.findall(pretok_pat, prepretok):
                    toks.extend(self._encode_pretok(pretok))

        return toks

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def _encode_pretok(self, pretok: str) -> list[int]:
        pretok_bytes = bytes(pretok, 'utf-8')
        toks = [bytes([b]) for b in pretok_bytes]
        for (left, right) in self.merges:
            if not (left in toks) and not (right in toks):
                continue
            mergedtoks = []
            it = peekable(toks)
            while (tok := next(it, None)) is not None:
                if tok == left:
                    next_tok = it.peek(None)
                    if next_tok == right:
                        mergedtoks.append(left + right)
                        # advance iterator to consume next token also
                        next(it, None)
                    else:
                        mergedtoks.append(tok)
                else:
                    mergedtoks.append(tok)
            toks = mergedtoks
        tokids = [self.inv_vocab[tok] for tok in toks]
        return tokids

    def decode(self, ids: list[int]) -> str:
        allbytes = b"".join((self.vocab[id] for id in ids))
        text = allbytes.decode(errors="replace")
        return text

