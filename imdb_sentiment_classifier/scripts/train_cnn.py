import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from ..bpe_tokenizer import BPETokenizer
from ..cnn import TextCNN
from ..data_utils import download_data
from ..dataset import IMDBDataset


class TextCNNLitModule(pl.LightningModule):
    def __init__(self, model: TextCNN, lr: float):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        loss = self.criterion(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)
        probs_pos = torch.softmax(logits, dim=-1)[:, 1]
        acc = (preds == batch["labels"]).float().mean()
        self.train_f1.update(preds, batch["labels"])
        self.train_auc.update(probs_pos, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_accuracy", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"])
        loss = self.criterion(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)
        probs_pos = torch.softmax(logits, dim=-1)[:, 1]
        acc = (preds == batch["labels"]).float().mean()
        self.val_f1.update(preds, batch["labels"])
        self.val_auc.update(probs_pos, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_accuracy", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        self.log(
            "train_f1",
            self.train_f1.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_auc",
            self.train_auc.compute(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.train_f1.reset()
        self.train_auc.reset()

    def on_validation_epoch_end(self):
        self.log(
            "val_f1",
            self.val_f1.compute(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_auc",
            self.val_auc.compute(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.val_f1.reset()
        self.val_auc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@hydra.main(config_path="pkg://configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    checkpoint_path = Path(to_absolute_path(cfg.paths.checkpoint_path))
    tokenizer_path = Path(to_absolute_path(cfg.paths.tokenizer_path))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    download_data()

    tokenizer = BPETokenizer(vocab_size=cfg.tokenizer.vocab_size)
    train_texts = []
    train_pos = Path(to_absolute_path(cfg.data.train.positive_path))
    train_neg = Path(to_absolute_path(cfg.data.train.negative_path))
    if not train_pos.exists() or not train_neg.exists():
        raise FileNotFoundError("Train data not found, expected IMDb train files.")
    for path in (train_pos, train_neg):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    train_texts.append(line)

    if tokenizer_path.exists():
        logging.info("Loading existing tokenizer from %s", tokenizer_path)
        tokenizer = BPETokenizer.load(str(tokenizer_path))
    else:
        logging.info(
            "Training BPE tokenizer on %d texts (num_merges=%d)",
            len(train_texts),
            cfg.tokenizer.num_merges,
        )
        tokenizer.train(train_texts, num_merges=cfg.tokenizer.num_merges)

    test_pos = Path(to_absolute_path(cfg.data.test.positive_path))
    test_neg = Path(to_absolute_path(cfg.data.test.negative_path))
    train_ds = IMDBDataset(
        str(train_pos),
        str(train_neg),
        tokenizer,
        max_length=cfg.data.max_length,
    )
    test_ds = IMDBDataset(
        str(test_pos),
        str(test_neg),
        tokenizer,
        max_length=cfg.data.max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False)
    logging.info(
        "Datasets ready: train=%d samples, val=%d samples, batch_size=%d",
        len(train_ds),
        len(test_ds),
        cfg.train.batch_size,
    )

    vocab_size = len(tokenizer.idx_to_token)
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=cfg.model.embedding_dim,
        num_filters=cfg.model.num_filters,
        dropout=cfg.model.dropout,
        num_classes=cfg.model.num_classes,
    )
    model.embedding.padding_idx = tokenizer.pad_token_id
    lit_module = TextCNNLitModule(model=model, lr=cfg.train.lr)
    logging.info(
        "Model initialized: vocab_size=%d, max_length=%d, lr=%g",
        vocab_size,
        cfg.data.max_length,
        cfg.train.lr,
    )

    mlflow_uri = cfg.mlflow.mlflow_uri or None
    logger = None
    if cfg.mlflow.enabled or mlflow_uri:
        logger = MLFlowLogger(
            experiment_name=cfg.mlflow.experiment,
            tracking_uri=mlflow_uri,
            run_name=cfg.mlflow.run_name,
        )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_path.parent),
        filename=checkpoint_path.stem,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=1,
    )

    logging.info("Starting training for %d epochs", cfg.train.epochs)
    trainer.fit(
        lit_module,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )

    torch.save(lit_module.model.state_dict(), checkpoint_path)
    print(f"Модель сохранена в {checkpoint_path}")
    tokenizer.save(str(tokenizer_path))
    print(f"Токенизатор сохранён в {tokenizer_path}")


if __name__ == "__main__":
    main()
