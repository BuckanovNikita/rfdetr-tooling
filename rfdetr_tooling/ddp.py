"""DDP launch helper — перезапуск через torchrun при gpus > 1."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from loguru import logger


def is_ddp_worker() -> bool:
    """Проверить, запущен ли процесс через torchrun (RANK и WORLD_SIZE в env)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _find_torchrun() -> str:
    """Поиск torchrun: сначала рядом с sys.executable, затем в PATH."""
    # torchrun обычно лежит в том же bin-каталоге что и python
    candidate = Path(sys.executable).parent / "torchrun"
    if candidate.is_file():
        return str(candidate)

    found = shutil.which("torchrun")
    if found is not None:
        return found

    logger.error("torchrun не найден. Установите PyTorch: pip install torch")
    sys.exit(1)


def relaunch_with_torchrun(nproc: int, argv: list[str]) -> NoReturn:
    """Перезапуск текущего CLI через torchrun для DDP.

    Строит команду:
        torchrun --standalone --nproc_per_node=N -m rfdetr_tooling.cli <argv>
    """
    torchrun = _find_torchrun()

    cmd = [
        torchrun,
        "--standalone",
        f"--nproc_per_node={nproc}",
        "-m",
        "rfdetr_tooling.cli",
        *argv,
    ]

    logger.info(f"DDP: перезапуск через torchrun ({nproc} GPU)")
    logger.debug(f"DDP cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)  # noqa: S603
    sys.exit(result.returncode)
