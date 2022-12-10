# Standard Library
import argparse
import os
import subprocess
from dataclasses import dataclass
from logging import NullHandler
from logging import getLogger
from pathlib import Path

# Third Party Library
import optuna

# First Party Library
from utils import NoStdoutException
from utils import ScoreSender

logger = getLogger(__name__)
logger.addHandler(NullHandler())


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="このプログラムの説明（なくてもよい）")
    parser.add_argument("--data_dir", required=True, type=lambda x: Path(x).expanduser().absolute())
    parser.add_argument("--out_dir", required=True, type=lambda x: Path(x).expanduser().absolute())
    parser.add_argument(
        "--data_filepath",
        type=lambda x: Path(x).expanduser().absolute(),
        help="'train_tf.txt' or 'test_tf.txt'",
    )
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    return args


@dataclass
class Cmd:
    cmd: list[str]
    cwd: Path | None = None


def run_cmd(cmd: Cmd) -> bytes:
    # subprocess.run(cmd.cmd, stdout=cmd.stdout, stderr=cmd.stderr, cwd=cmd.cwd)
    with subprocess.Popen(cmd.cmd, stdout=subprocess.PIPE, cwd=cmd.cwd) as proc:
        if proc.stdout is None:
            raise NoStdoutException("proc.stdout is None")
        stdout, _stderr = proc.communicate()
        return stdout


def main() -> None:
    # args = get_args()
    def objective(trial: optuna.trial.Trial) -> float:
        param1 = trial.suggest_float("param1", -10.0, 10.0)
        print(param1)

        cmd = Cmd(
            cmd=[
                "poetry",
                "run",
                "python",
                "main.py",
            ],
            cwd=None,
        )

        score = ScoreSender.extract_score(ScoreSender.extract_path_from_stdout(run_cmd(cmd)))

        return score

    pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner()  # optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        study_name="sample-study",
        sampler=sampler,
        direction=optuna.study.StudyDirection.MAXIMIZE,
        storage="sqlite:///db.sqlite3",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=5)

    trial = study.best_trial

    print("Accuracy: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))


if __name__ == "__main__":
    # Standard Library
    import logging

    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
        level=logging.WARNING,
    )

    main()
