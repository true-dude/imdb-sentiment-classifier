"""
Конвертирует Kaggle IMDB Dataset (50k reviews) в формат
data/IMDb/train|test/{positive,negative}.txt.
Ожидается CSV по пути data/kaggle_imdb/IMDB Dataset.csv (review,sentiment).
"""

import csv
from pathlib import Path


def main() -> None:
    src = Path("data/kaggle_imdb/IMDB Dataset.csv")
    if not src.exists():
        raise FileNotFoundError(
            f"Не найден {src}. Скачайте датасет с Kaggle и распакуйте CSV сюда."
        )

    out_root = Path("data/IMDb")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with src.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["review"].replace("\n", " ").strip()
            label = 1 if row["sentiment"].lower().startswith("pos") else 0
            rows.append((text, label))

    split = int(len(rows) * 0.8)
    train_rows, test_rows = rows[:split], rows[split:]

    def dump(split_name: str, data):
        pos = out_root / split_name / "positive.txt"
        neg = out_root / split_name / "negative.txt"
        pos.parent.mkdir(parents=True, exist_ok=True)
        with (
            pos.open("w", encoding="utf-8") as fp,
            neg.open("w", encoding="utf-8") as fn,
        ):
            for text, label in data:
                (fp if label == 1 else fn).write(text + "\n")

    dump("train", train_rows)
    dump("test", test_rows)
    print("IMDB txt файлы готовы в data/IMDb/train|test/{positive,negative}.txt")


if __name__ == "__main__":
    main()
