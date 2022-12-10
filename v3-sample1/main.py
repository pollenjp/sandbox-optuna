# Standard Library
import typing as t
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from dataclasses import dataclass
from pathlib import Path

# Third Party Library
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf


def assert_sequence_config(cfg: MutableSequence[t.Any]) -> None:
    for val in cfg:
        if isinstance(val, MutableMapping):
            assert_mapping_config(val)
        elif isinstance(val, MutableSequence):
            assert_sequence_config(val)


def assert_mapping_config(cfg: MutableMapping[str, t.Any]) -> None:
    for _, val in cfg.items():
        if isinstance(val, MutableMapping):
            assert_mapping_config(val)
        elif isinstance(val, MutableSequence):
            assert_sequence_config(val)


@dataclass
class Config:
    log_root_path: str
    param1: float
    param2: float


@hydra.main(
    version_base=None,
    config_path=f"{Path(__file__).parents[0] / 'conf'}",
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    cfg = t.cast(DictConfig, OmegaConf.merge(OmegaConf.structured(Config), cfg))
    assert_mapping_config(cfg)
    config = t.cast(Config, cfg)
    del cfg

    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    main()
