import logging
import shutil
import subprocess
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Config (export TensorRT):\n%s", OmegaConf.to_yaml(cfg))

    onnx_path = Path(to_absolute_path(cfg.paths.onnx_path))
    trt_path = Path(to_absolute_path(cfg.paths.trt_path))

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX-модель {onnx_path} не найдена. Сначала запусти export_onnx.py."
        )

    trtexec = shutil.which("trtexec")
    if trtexec is None:  # pragma: no cover - depends on env
        raise SystemExit(
            "Не найден trtexec. Установи TensorRT и убедись, что trtexec в PATH."
        )

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        "--minShapes=input_ids:1x8",
        f"--optShapes=input_ids:1x{cfg.data.max_length}",
        f"--maxShapes=input_ids:4x{cfg.data.max_length}",
        "--fp16",
    ]

    logging.info("Запускаю trtexec: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logging.info("TensorRT движок сохранён в %s", trt_path)
    print(f"TensorRT движок сохранён в {trt_path}")


if __name__ == "__main__":
    main()
