import csv
import logging
import subprocess
from pathlib import Path
from typing import Optional


def _prepare_imdb_from_csv(csv_path: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    train_pos = out_root / "train" / "positive.txt"
    train_neg = out_root / "train" / "negative.txt"
    test_pos = out_root / "test" / "positive.txt"
    test_neg = out_root / "test" / "negative.txt"
    train_pos.parent.mkdir(parents=True, exist_ok=True)
    test_pos.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["review"].replace("\n", " ").strip()
            label = 1 if row["sentiment"].lower().startswith("pos") else 0
            rows.append((text, label))
    split = int(len(rows) * 0.8)
    train_rows, test_rows = rows[:split], rows[split:]

    with (
        train_pos.open("w", encoding="utf-8") as fp,
        train_neg.open("w", encoding="utf-8") as fn,
    ):
        for text, label in train_rows:
            (fp if label == 1 else fn).write(text + "\n")

    with (
        test_pos.open("w", encoding="utf-8") as fp,
        test_neg.open("w", encoding="utf-8") as fn,
    ):
        for text, label in test_rows:
            (fp if label == 1 else fn).write(text + "\n")


def download_data(kaggle_csv: Optional[str] = None) -> None:
    """
    Ensure data/IMDb exists. Tries DVC first, then local Kaggle CSV if provided.
    """
    target = Path("data/IMDb/train/positive.txt")
    if target.exists():
        return

    try:
        logging.info("Trying to fetch data via DVC (pull)")
        try:
            import os

            remote_name = os.getenv("DVC_REMOTE")
        except Exception:
            remote_name = None
        pull_cmd = ["dvc", "pull", "data/IMDb.dvc"]
        if remote_name:
            pull_cmd.extend(["-r", remote_name])
        subprocess.run(pull_cmd, check=True)
        if target.exists():
            return
    except Exception as e:
        logging.warning("dvc pull failed: %s", e)

    dvc_file = Path("data/IMDb.dvc")
    if dvc_file.exists():
        try:
            subprocess.run(["dvc", "pull", str(dvc_file)], check=True)
            if target.exists():
                return
        except Exception as e:
            logging.warning("dvc pull failed: %s", e)

    csv_path = (
        Path(kaggle_csv) if kaggle_csv else Path("data/kaggle_imdb/IMDB Dataset.csv")
    )
    if csv_path.exists():
        logging.info("Building IMDb txt files from %s", csv_path)
        _prepare_imdb_from_csv(csv_path, Path("data/IMDb"))
        return

    raise FileNotFoundError(
        "Данных нет. Выполните dvc pull или положите Kaggle CSV в "
        "data/kaggle_imdb/IMDB Dataset.csv"
    )
