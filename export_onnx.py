import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from bpe_tokenizer import BPETokenizer
from CNN import TextCNN


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Config (export ONNX):\n%s", OmegaConf.to_yaml(cfg))

    try:
        import onnx  # type: ignore
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover - informative path
        raise SystemExit(
            "Нужны пакеты onnx и onnxruntime. Установи их, например: "
            "uv pip install onnx onnxruntime"
        ) from exc

    checkpoint_path = Path(to_absolute_path(cfg.paths.checkpoint_path))
    tokenizer_path = Path(to_absolute_path(cfg.paths.tokenizer_path))
    onnx_path = Path(to_absolute_path(cfg.paths.onnx_path))

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} не найден. Сначала запусти train_cnn.py."
        )
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Токенизатор {tokenizer_path} не найден. Сначала запусти train_cnn.py."
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
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    seq_len = cfg.data.max_length
    dummy_input = torch.zeros((1, seq_len), dtype=torch.long)

    logging.info("Exporting ONNX to %s", onnx_path)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        # Статическая длина (seq_len из конфига) во избежание ошибок dynamo
        dynamo=False,
        opset_version=18,
    )

    onnx.checker.check_model(onnx.load(onnx_path))
    session = ort.InferenceSession(onnx_path)
    outputs = session.run(None, {"input_ids": dummy_input.numpy()})
    logging.info("ONNX готов: вывод формы %s", outputs[0].shape)
    print(f"ONNX модель сохранена в {onnx_path}")


if __name__ == "__main__":
    main()
