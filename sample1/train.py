# Standard Library
import argparse
import os
from typing import Callable
from typing import List
from typing import Optional

# Third Party Library
import optuna
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms

if version.parse(pl.__version__) < version.parse("1.0.2"):
    raise RuntimeError("PyTorch Lightning>=1.0.2 is required for this example.")


class Net(nn.Module):
    def __init__(self, dropout: float, output_dims: List[int], num_classes: int):
        super().__init__()
        layers: List[nn.Module] = []

        input_dim: int = 28 * 28
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, num_classes))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return F.log_softmax(logits, dim=1)


class LightningNet(pl.LightningModule):
    def __init__(self, dropout: float, output_dims: List[int], num_classes: int):
        super().__init__()
        self.model = Net(dropout, output_dims, num_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data.view(-1, 28 * 28))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        return F.nll_loss(output, target)

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters())


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers: int = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.mnist_test = datasets.FashionMNIST(
            self.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        mnist_full = datasets.FashionMNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True
        )


def create_objective() -> Callable[[optuna.trial.Trial], float]:
    PERCENT_VALID_EXAMPLES = 0.1
    BATCHSIZE = 128
    CLASSES = 10
    EPOCHS = 10
    DIR = os.getcwd()
    NUM_WORKERS: int = 2

    def objective(trial: optuna.trial.Trial) -> float:

        # We optimize the number of layers, hidden units in each layer and dropouts.
        n_layers = trial.suggest_int("n_layers", 1, 5)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        output_dims = [trial.suggest_int("n_units_l{}".format(i), 4, 256, log=True) for i in range(n_layers)]

        model = LightningNet(dropout, output_dims, num_classes=CLASSES)
        datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE, num_workers=NUM_WORKERS)

        trainer = pl.Trainer(
            logger=True,
            limit_val_batches=PERCENT_VALID_EXAMPLES,
            checkpoint_callback=False,
            max_epochs=EPOCHS,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
        )
        hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        return trainer.callback_metrics["val_acc"].item()

    return objective


if __name__ == "__main__":
    pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner()  # optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        study_name="my-study",
        sampler=sampler,
        direction=optuna.study.StudyDirection.MAXIMIZE,
        storage="sqlite:///my-storage.db",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(create_objective(), n_trials=200, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
