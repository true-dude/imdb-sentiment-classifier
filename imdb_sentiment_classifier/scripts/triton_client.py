import json
from typing import List

import fire
import requests

from ..bpe_tokenizer import BPETokenizer


def prepare_ids(tokenizer: BPETokenizer, text: str, max_length: int) -> List[int]:
    encoded = tokenizer.encode(text, return_attention_mask=False)["input_ids"][
        :max_length
    ]
    pad_id = tokenizer.pad_token_id
    if len(encoded) < max_length:
        encoded += [pad_id] * (max_length - len(encoded))
    return encoded


def main(
    text: str,
    tokenizer_path: str = "artifacts/tokenizer.json",
    url: str = "http://localhost:8000/v2/models/textcnn/infer",
    max_length: int = 128,
    timeout: float = 10.0,
) -> None:
    tokenizer = BPETokenizer.load(tokenizer_path)
    input_ids = prepare_ids(tokenizer, text, max_length)

    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, max_length],
                "datatype": "INT64",
                "data": input_ids,
            }
        ],
        "outputs": [{"name": "logits"}],
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    logits = data["outputs"][0]["data"]
    pred = int(logits.index(max(logits))) if logits else None
    labels = {0: "negative", 1: "positive"}

    print("Response:", json.dumps(data, indent=2))
    if pred is not None:
        print(f"Predicted class: {labels.get(pred, pred)}")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
