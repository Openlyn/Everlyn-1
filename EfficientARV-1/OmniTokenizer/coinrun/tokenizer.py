# take from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
# to give users a quick easy start to training DALL-E without doing BPE

import torch

import html
import os
from functools import lru_cache
from pathlib import Path
import ftfy
import regex as re


# OpenAI simple tokenizer

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "language_model/bpe_simple_vocab_16e6.txt")
    # return "./language_model/bpe_simple_vocab_16e6.txt"


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path=default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = Path(bpe_path).read_text(encoding='utf8').split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])

        self.vocab_size = 49408

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens, remove_start_end=True):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()

        if remove_start_end:
            tokens = [token for token in tokens if token not in (49406, 40407, 0)]
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def tokenize(self, texts, context_length=256, truncate_text=False):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        # all_tokens = [self.encode(text) for text in texts]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result



    def tokenize_future(self, texts, future_texts, context_length=256, truncate_text=False):
        if isinstance(texts, str):
            texts = [texts]

        if isinstance(future_texts, str):
            future_texts = [future_texts]

        all_tokens = [self.encode(text) for text in texts]
        all_future_tokens = [self.encode(future_text) for future_text in future_texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        token_types = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, (tokens, future_tokens) in enumerate(zip(all_tokens, all_future_tokens)):
            if len(tokens) + len(future_tokens) > context_length:
                if truncate_text:
                    if len(tokens) > context_length:
                        tokens = tokens[:context_length]
                        future_tokens = []
                    else:
                        future_tokens = future_tokens[:context_length-len(tokens)]
                else:
                    raise RuntimeError(f"Input {texts[i]} and {future_tokens[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
            result[i, len(tokens):len(tokens)+len(future_tokens)] = torch.tensor(future_tokens)
            token_types[i, :len(tokens)] = 1

        return result, token_types

tokenizer = SimpleTokenizer()
