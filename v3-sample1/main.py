# Standard Library
import random
import typing as t
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from dataclasses import dataclass
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

# First Party Library
from utils import ScoreSender

logger = getLogger(__name__)
logger.addHandler(NullHandler())


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

    logger.info(OmegaConf.to_yaml(config))

    log_root_path = Path(config.log_root_path).expanduser()

    # TODO: fit model

    # save score
    score = random.random()
    ScoreSender.save_score(log_root_path, score)


if __name__ == "__main__":
    # Standard Library
    import logging

    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
        level=logging.WARNING,
    )
    logger.setLevel(logging.INFO)

    main()
