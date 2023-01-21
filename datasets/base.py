import copy
import string
from abc import ABC
from collections.abc import Mapping, Sequence

from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.core.datamodule import \
    LightningDataModule as _LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset

from utils import deep_update


class LightningDataModule(_LightningDataModule):
    SPLIT_NAMES = ["train", "val", "test", "predict"]

    def __init__(
        self,
        dataset_cfg: dict,
        dataloader_cfg=None,
        split_format_to="ann_file",
        split_name_map=None,
    ):
        super().__init__()

        self.dataset_cfg = self.get_split_config(dataset_cfg)
        self.dataloader_cfg = self.get_split_config(dataloader_cfg)

        self.split_format_to = (
            split_format_to
            if split_format_to is None or isinstance(split_format_to, list)
            else [split_format_to]
        )

        for split in self.SPLIT_NAMES:
            if self.dataset_cfg[split].get("init_args") is None:
                self.dataset_cfg[split]["init_args"] = {}
            if self.split_format_to is not None:
                for s in self.split_format_to:
                    self.dataset_cfg[split]["init_args"][s] = string.Template(
                        self.dataset_cfg[split]["init_args"].get(s, "$split")
                    ).safe_substitute(split=split)

        if split_name_map is None:
            self.split_name_map = {}
        else:
            self.split_name_map = split_name_map
        for name in self.SPLIT_NAMES:
            self.split_name_map.setdefault(name, name if name != "predict" else "test")

        self.datasets = {}
        self.dataset = None

    def get_split_config(self, config):
        res = {}
        if isinstance(config, Mapping):
            if all([config.get(name) is None for name in self.SPLIT_NAMES]):
                res = {name: copy.deepcopy(config) for name in self.SPLIT_NAMES}
            else:
                res = {}
                for main_name in self.SPLIT_NAMES:
                    if config.get(main_name) is not None:
                        break
                for name in self.SPLIT_NAMES:
                    if config.get(name) is None or name == main_name:
                        res[name] = copy.deepcopy(config[main_name])
                    else:
                        res[name] = deep_update(
                            copy.deepcopy(config[main_name]), config[name]
                        )
        return res

    def _get_split_names(self, stage=None):
        if self.trainer.overfit_batches > 0:
            split_names = ["train"]
        elif stage is None:
            split_names = ["train", "val", "test", "predict"]
        elif stage == "fit":
            split_names = ["train", "val"]
        else:
            split_names = [stage.lower()]
        return split_names

    def _setup_dataset(self, split_name):
        self.datasets[split_name] = self._build_data_set(
            self.split_name_map[split_name]
        )

    def setup(self, stage=None):
        split_names = self._get_split_names(stage)

        for name in split_names:
            self._setup_dataset(name)
        self.dataset = self.datasets[split_names[0]]

    def _dataloader(self, split_name, **kwargs):
        return self._build_data_loader(
            self.datasets[split_name], split=split_name, **kwargs
        )

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("val")

    def test_dataloader(self):
        return self._dataloader("test")

    def predict_dataloader(self):
        return self._dataloader("predict")

    def _build_data_set(self, split):
        return instantiate_class(tuple(), self.dataset_cfg[split])

    def _build_data_loader(self, dataset, split="train"):
        if isinstance(dataset, Mapping):
            return {
                key: self._build_data_loader(
                    ds,
                    split=split,
                )
                for key, ds in dataset.items()
            }
        elif isinstance(dataset, Sequence):
            return [
                self._build_data_loader(
                    ds,
                    split=split,
                )
                for ds in dataset
            ]
        else:
            collate_fn = self.collate_fn if hasattr(self, "collate_fn") else None
            return DataLoader(
                dataset=dataset, collate_fn=collate_fn, **self.dataloader_cfg[split]
            )


class KFoldLightningDataModule(LightningDataModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_folds = None
        self.folds = {}
        self.splits = []

    def _get_split_names(self, stage=None):
        if self.trainer.overfit_batches > 0:
            split_names = ["train"]
        elif stage is None:
            split_names = ["train", "val", "test", "predict"]
        elif stage == "fit":
            split_names = ["train"]
        else:
            split_names = [stage.lower()]
        return split_names

    def setup(self, stage=None):
        super().setup(stage)
        split_names = self._get_split_names(stage)
        if "train" in split_names or "val" in split_names:
            self.setup_folds(2)
            self.setup_fold_index(0)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [
            split for split in KFold(num_folds).split(range(len(self.dataset)))
        ]

    def setup_fold_index(self, fold_index: int) -> None:
        for indices, fold_name in zip(self.splits[fold_index], ["train", "val"]):
            self.folds[fold_name] = Subset(self.dataset, indices)

    def _fold_dataloader(self, split_name, **kwargs):
        return self._build_data_loader(
            self.folds[split_name], split=split_name, **kwargs
        )

    def train_dataloader(self):
        return self._fold_dataloader("train")

    def val_dataloader(self):
        return self._fold_dataloader("val")
