"""CLI entry point для rfdetr-tooling.

Синтаксис: rfdetr-tool <command> [key=value ...]
Команды: train, val, predict, cfg, test
"""

from __future__ import annotations

import sys
import types
import typing
from pathlib import Path
from typing import Any, get_type_hints

import yaml
from loguru import logger
from pydantic import BaseModel, ValidationError

from rfdetr_tooling.config import (
    VARIANT_RESOLUTION,
    PredictConfig,
    TrainConfig,
    ValConfig,
)

_COMMANDS: dict[str, type[BaseModel]] = {
    "train": TrainConfig,
    "val": ValConfig,
    "predict": PredictConfig,
}


def _print_help() -> None:
    """Вывод справки по командам и параметрам."""
    msg = """\
rfdetr-tool — CLI для тренировки, валидации и инференса RF-DETR

Использование:
  rfdetr-tool <command> [key=value ...]

Команды:
  train    Тренировка модели
  val      Валидация на val-сете (mAP)
  predict  Инференс на изображениях
  cfg      Генерация дефолтного YAML-конфига
  test     Smoke-тесты CLI, линтеров и mypy

Примеры:
  rfdetr-tool train data=./dataset variant=base epochs=100 batch_size=8
  rfdetr-tool train cfg=config.yaml data=./dataset epochs=50
  rfdetr-tool train data=./dataset clearml=true project=my-project run=exp-1
  rfdetr-tool predict source=./images weights=model.pth
  rfdetr-tool predict source=./dataset weights=model.pth format=csv
  rfdetr-tool predict source=dir1,dir2,data.yaml weights=model.pth
  rfdetr-tool predict source=./img weights=m.pth conf_threshold=0.25
  rfdetr-tool predict source=./img weights=m.pth batch_size=16 device=cuda:1
  rfdetr-tool predict source=./images weights=model.pth visualize=true
  rfdetr-tool val weights=output/checkpoint.pth data=./dataset
  rfdetr-tool cfg variant=base
  rfdetr-tool cfg variant=large output=large_config.yaml

Приоритет конфигов: CLI > YAML (cfg=path.yaml) > pydantic defaults
"""
    sys.stdout.write(msg)


def _parse_argv(argv: list[str]) -> tuple[str, dict[str, str]]:
    """Извлекает команду и dict {key: raw_string} из argv."""
    if not argv:
        _print_help()
        sys.exit(0)

    command = argv[0]
    overrides: dict[str, str] = {}
    for arg in argv[1:]:
        if "=" not in arg:
            logger.error(f"Неверный аргумент: {arg!r}. Ожидается формат key=value")
            sys.exit(1)
        key, value = arg.split("=", 1)
        overrides[key] = value

    return command, overrides


def _coerce_value(value: str, annotation: Any) -> Any:  # noqa: ANN401, PLR0911
    """Приведение строкового значения к типу по аннотации pydantic-поля."""
    origin = typing.get_origin(annotation)

    # Обработка Union / Optional (typing.Union и types.UnionType)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        args = typing.get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        # Если Union содержит tuple — не приводить, пусть pydantic разбирается
        if any(typing.get_origin(a) is tuple or a is tuple for a in non_none):
            return value
        if non_none:
            return _coerce_value(value, non_none[0])

    # Обработка Literal
    if origin is typing.Literal:
        return value

    if annotation is bool:
        return value.lower() in ("true", "1", "yes")
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    return value


def _build_config(
    config_cls: type[BaseModel],
    overrides: dict[str, str],
) -> BaseModel:
    """Загружает YAML (если cfg=), накладывает CLI-переопределения, создаёт модель."""
    data: dict[str, Any] = {}

    # Загрузка YAML если указан cfg=
    cfg_path = overrides.pop("cfg", None)
    if cfg_path is not None:
        path = Path(cfg_path)
        if not path.exists():
            logger.error(f"Файл конфига не найден: {cfg_path}")
            sys.exit(1)
        with path.open() as f:
            yaml_data = yaml.safe_load(f)
        if isinstance(yaml_data, dict):
            data.update(yaml_data)

    # CLI-переопределения поверх YAML
    hints = get_type_hints(config_cls)
    for key, raw_value in overrides.items():
        if key not in hints:
            logger.error(
                f"Неизвестный параметр {key!r} для {config_cls.__name__}. "
                f"Допустимые: {sorted(hints)}"
            )
            sys.exit(1)
        data[key] = _coerce_value(raw_value, hints[key])

    try:
        return config_cls(**data)
    except ValidationError as exc:
        logger.error(f"Ошибка валидации конфигурации:\n{exc}")
        sys.exit(1)


def _cmd_cfg(overrides: dict[str, str]) -> None:
    """Генерация дефолтного YAML-конфига."""
    variant = overrides.pop("variant", "base")
    output = overrides.pop("output", None)

    # Передаём оставшиеся overrides (resolution и др.) в TrainConfig
    extra: dict[str, Any] = {}
    hints = get_type_hints(TrainConfig)
    for key, raw_value in overrides.items():
        if key in hints:
            extra[key] = _coerce_value(raw_value, hints[key])

    config = TrainConfig(data="./dataset", variant=variant, **extra)  # type: ignore[arg-type]

    # Подставляем дефолтный resolution для варианта если не задан явно
    if config.resolution is None:
        config = config.model_copy(
            update={"resolution": VARIANT_RESOLUTION.get(variant, 560)}
        )

    data = config.model_dump()

    # Сериализация tuple resolution обратно в WxH строку
    if isinstance(data.get("resolution"), tuple):
        h, w = data["resolution"]
        data["resolution"] = f"{w}x{h}"

    # YAML с комментариями по секциям
    sections = {
        "# --- Датасет ---": ["data", "variant"],
        "\n# --- Обучение ---": [
            "epochs",
            "batch_size",
            "lr",
            "lr_encoder",
            "lr_drop",
            "weight_decay",
            "grad_accum_steps",
            "warmup_epochs",
        ],
        "\n# --- EMA ---": ["use_ema", "ema_decay", "ema_tau"],
        "\n# --- Early stopping ---": [
            "early_stopping",
            "early_stopping_patience",
            "early_stopping_min_delta",
        ],
        "\n# --- Выходные данные ---": [
            "output_dir",
            "device",
            "resume",
            "checkpoint_interval",
        ],
        "\n# --- Логирование ---": [
            "tensorboard",
            "wandb",
            "mlflow",
            "clearml",
            "project",
            "run",
        ],
        "\n# --- Распределённое обучение (DDP) ---": ["gpus", "sync_bn"],
        "\n# --- Прочее ---": [
            "gradient_checkpointing",
            "drop_path",
            "seed",
            "num_workers",
            "multi_scale",
            "resolution",
            "progress_bar",
        ],
    }

    lines: list[str] = []
    for comment, keys in sections.items():
        lines.append(comment)
        section_data = {k: data[k] for k in keys if k in data}
        if section_data:
            lines.append(yaml.dump(section_data, default_flow_style=False).rstrip())

    content = "\n".join(lines) + "\n"

    if output:
        Path(output).write_text(content)
        logger.info(f"Конфиг записан в {output}")
    else:
        sys.stdout.write(content)


def _cmd_train(config: TrainConfig, argv: list[str]) -> None:
    """Запуск тренировки с поддержкой DDP."""
    from rfdetr_tooling.ddp import (  # noqa: PLC0415
        is_ddp_worker,
        relaunch_with_torchrun,
    )
    from rfdetr_tooling.train import train  # noqa: PLC0415

    # Preflight: gpus > 1 несовместимо с cpu/mps
    if config.gpus > 1 and config.device in ("cpu", "mps"):
        logger.error(
            f"DDP (gpus={config.gpus}) несовместим с device={config.device!r}"  # noqa: RUF001
        )
        sys.exit(1)

    # Preflight: проверка доступных GPU
    if config.gpus > 1:
        try:
            import torch  # noqa: PLC0415

            available = torch.cuda.device_count()
            if config.gpus > available:
                logger.error(
                    f"Запрошено gpus={config.gpus}, но доступно только {available} GPU"
                )
                sys.exit(1)
        except ImportError:
            pass  # torch не установлен — torchrun сам покажет ошибку

    # DDP relaunch
    if config.gpus > 1 and not is_ddp_worker():
        relaunch_with_torchrun(config.gpus, argv)

    train(**config.model_dump())


def main(argv: list[str] | None = None) -> None:
    """Entry point для CLI."""
    if argv is None:
        argv = sys.argv[1:]

    command, overrides = _parse_argv(argv)

    if command == "cfg":
        _cmd_cfg(overrides)
        return

    if command == "test":
        from rfdetr_tooling.test_runner import run_tests  # noqa: PLC0415

        run_tests(overrides)
        return

    if command not in _COMMANDS:
        logger.error(
            f"Неизвестная команда: {command!r}. "
            f"Допустимые: {', '.join([*_COMMANDS, 'cfg', 'test'])}"
        )
        sys.exit(1)

    config_cls = _COMMANDS[command]
    config = _build_config(config_cls, overrides)

    if command == "train":
        assert isinstance(config, TrainConfig)  # noqa: S101
        _cmd_train(config, argv)
    elif command == "predict":
        from rfdetr_tooling.predict import predict  # noqa: PLC0415

        predict(**config.model_dump())
    elif command == "val":
        from rfdetr_tooling.val import val  # noqa: PLC0415

        val(**config.model_dump())
