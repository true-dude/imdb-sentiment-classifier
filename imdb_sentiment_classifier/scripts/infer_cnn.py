import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ..bpe_tokenizer import BPETokenizer
from ..cnn import TextCNN
from ..data_utils import download_data


@hydra.main(config_path="pkg://configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Config (inference):\n%s", OmegaConf.to_yaml(cfg))

    checkpoint_path = Path(to_absolute_path(cfg.paths.checkpoint_path))
    tokenizer_path = Path(to_absolute_path(cfg.paths.tokenizer_path))

    download_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Токенизатор {tokenizer_path} не найден. "
            "Сначала запусти train_cnn.py, чтобы сохранить его."
        )
    tokenizer = BPETokenizer.load(str(tokenizer_path))

    model = TextCNN(
        vocab_size=len(tokenizer.idx_to_token),
        embedding_dim=cfg.model.embedding_dim,
        num_filters=cfg.model.num_filters,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
    )
    model.embedding.padding_idx = tokenizer.pad_token_id

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} не найден, сначала запусти train_cnn.py"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    text = cfg.get("text")
    if not text:
        raise ValueError("Укажите текст для инференса: text='...'.")

    encoded = tokenizer.encode(text, return_attention_mask=True)
    input_ids = encoded["input_ids"][: cfg.data.max_length]
    if len(input_ids) < cfg.data.max_length:
        pad_id = tokenizer.pad_token_id
        input_ids = input_ids + [pad_id] * (cfg.data.max_length - len(input_ids))

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_class = int(torch.argmax(probs).item())

    labels = {0: "negative", 1: "positive"}
    print(f"Текст: {text}")
    print(
        f"Класс: {labels.get(pred_class, pred_class)} "
        f"(p={probs[pred_class].item():.3f})"
    )


if __name__ == "__main__":
    main()
