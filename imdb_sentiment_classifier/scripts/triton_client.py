import argparse
import json
from typing import List

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple Triton HTTP client for TextCNN ONNX model"
    )
    parser.add_argument("--text", required=True, help="Текст для инференса")
    parser.add_argument(
        "--tokenizer-path",
        default="artifacts/tokenizer.json",
        help="Путь к сохранённому BPETokenizer",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v2/models/textcnn/infer",
        help="HTTP endpoint Triton",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Длина, до которой паддятся input_ids",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Таймаут HTTP-запроса, секунд"
    )
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer_path)
    input_ids = prepare_ids(tokenizer, args.text, args.max_length)

    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, args.max_length],
                "datatype": "INT64",
                "data": input_ids,
            }
        ],
        "outputs": [{"name": "logits"}],
    }

    resp = requests.post(args.url, json=payload, timeout=args.timeout)
    resp.raise_for_status()
    data = resp.json()

    logits = data["outputs"][0]["data"]
    pred = int(logits.index(max(logits))) if logits else None
    labels = {0: "negative", 1: "positive"}

    print("Response:", json.dumps(data, indent=2))
    if pred is not None:
        print(f"Predicted class: {labels.get(pred, pred)}")


if __name__ == "__main__":
    main()
