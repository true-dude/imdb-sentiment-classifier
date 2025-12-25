import json
import re
from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.idx_to_token = {}
        self.merges = {}
        self.token_to_idx = {}
        self.special_tokens = {}
        self.unk_token = "<unk>"
        self.w_token = "</w>"
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.special_tokens = {
            self.unk_token: 1,
            self.pad_token: 0,
            self.w_token: 2,
        }
        self.pat = re.compile(r"\w+[\w'-]*|\S")
        self.idx_to_token = {i: chr(i) for i in range(256)}
        self.idx_to_token.update(
            {idx: token for token, idx in self.special_tokens.items()}
        )
        self.token_to_idx = {token: idx for idx, token in self.idx_to_token.items()}

    def train(self, texts, num_merges):
        words = []
        for text in texts:
            tokens = re.findall(self.pat, text)
            words.extend([list(token) + ["</w>"] for token in tokens])

        token_freqs = Counter()
        for word in words:
            token_freqs.update([tuple(word)])
        for _ in range(num_merges):
            if len(self.idx_to_token) >= self.vocab_size:
                break
            pair_counts = defaultdict(int)
            for tokens, count in token_freqs.items():
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += count
            if not pair_counts:
                break

            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            new_token = "".join(best_pair)

            new_id = len(self.idx_to_token)
            self.idx_to_token[new_id] = new_token
            self.token_to_idx[new_token] = new_id
            self.merges[best_pair] = new_token

            token_freqs = self._updating_vocab(best_pair, new_token, token_freqs)

    def _updating_vocab(self, pair, new_token, token_freqs):
        new_token_freqs = Counter()
        for tokens, count in token_freqs.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_token_freqs[tuple(new_tokens)] += count
        return new_token_freqs

    def encode(self, text, return_attention_mask):
        tokens = self.tokenize(text)
        input_ids = [
            self.token_to_idx.get(token, self.special_tokens[self.unk_token])
            for token in tokens
        ]

        output = {"input_ids": input_ids}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(input_ids)
        return output

    def tokenize(self, text):
        words = re.findall(self.pat, text)
        tokens = []
        for word in words:
            current_tokens = list(word) + ["</w>"]
            while len(current_tokens) > 1:
                pairs = list(zip(current_tokens[:-1], current_tokens[1:]))
                best_pair = None
                for pair in pairs:
                    if pair in self.merges:
                        best_pair = pair
                        break
                if not best_pair:
                    break

                merged_token = self.merges[best_pair]
                new_tokens = []
                i = 0
                while i < len(current_tokens):
                    if (
                        i < len(current_tokens) - 1
                        and (current_tokens[i], current_tokens[i + 1]) == best_pair
                    ):
                        new_tokens.append(merged_token)
                        i += 2
                    else:
                        new_tokens.append(current_tokens[i])
                        i += 1
                current_tokens = new_tokens
            tokens.extend(current_tokens)
        return tokens

    def decode(self, input_ids):
        tokens = [
            self.idx_to_token.get(token_id, self.unk_token) for token_id in input_ids
        ]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    def save(self, path: str) -> None:
        merges_list = [
            [pair[0], pair[1], merged] for pair, merged in self.merges.items()
        ]
        state = {
            "vocab_size": self.vocab_size,
            "idx_to_token": {int(k): v for k, v in self.idx_to_token.items()},
            "token_to_idx": self.token_to_idx,
            "merges": merges_list,
            "special_tokens": self.special_tokens,
            "unk_token": self.unk_token,
            "w_token": self.w_token,
            "pad_token": self.pad_token,
            "pad_token_id": self.pad_token_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        tokenizer = cls(vocab_size=state.get("vocab_size", 30000))

        tokenizer.idx_to_token = {int(k): v for k, v in state["idx_to_token"].items()}
        tokenizer.token_to_idx = {
            k: int(v) if isinstance(v, str) and v.isdigit() else v
            for k, v in state["token_to_idx"].items()
        }

        tokenizer.merges = {}
        for src_a, src_b, merged in state.get("merges", []):
            tokenizer.merges[(src_a, src_b)] = merged

        tokenizer.special_tokens = state.get("special_tokens", tokenizer.special_tokens)
        tokenizer.unk_token = state.get("unk_token", tokenizer.unk_token)
        tokenizer.w_token = state.get("w_token", tokenizer.w_token)
        tokenizer.pad_token = state.get("pad_token", tokenizer.pad_token)
        tokenizer.pad_token_id = state.get("pad_token_id", tokenizer.pad_token_id)

        return tokenizer
