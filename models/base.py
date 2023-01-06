from abc import ABC
from typing import Mapping

import torch
from lightning.pytorch import LightningModule as _LightningModule
from mmengine.model import BaseModule


class LightningModule(_LightningModule, BaseModule, ABC):
    def __init__(
        self,
        normalize_config=None,
        loss_weights=None,
        loss_modules: Mapping[str, torch.nn.Module] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.automatic_lr_schedule = True
        self.normalize_config = normalize_config
        self.lr = None
        self.batch_size = None
        self.loss_weights = loss_weights
        self.loss_modules = torch.nn.ModuleDict(loss_modules if loss_modules else {})

    def configure_optimizer_parameters(self):
        return None

    @staticmethod
    def add_prefix(log_dict, prefix="train", sep="/"):
        return {f"{prefix}{sep}{k}": v for k, v in log_dict.items()}

    def log(self, *args, batch_size=None, **kwargs):
        if (
            batch_size is None
            and hasattr(self, "batch_size")
            and self.batch_size is not None
        ):
            batch_size = self.batch_size
        super().log(*args, batch_size=batch_size, **kwargs)

    def _loss_step(self, batch, res):
        raise NotImplementedError

    def loss_step(
        self,
        batch,
        res,
        use_loss_weight=True,
    ):
        loss = self._loss_step(batch, res)
        # multi loss weights
        if use_loss_weight and self.loss_weights:
            loss = {
                k: v * (1 if k not in self.loss_weights else self.loss_weights[k])
                for k, v in loss.items()
            }
        # calculate loss
        if 'loss' not in loss:
            loss["loss"] = torch.sum(torch.stack(list(loss.values())))
        return loss

    def on_fit_start(self):
        self.init_weights()

    def _dump_init_info(self, *args, **kwargs):
        pass

    def training_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss_dict = self.loss_step(batch, res)
        self.log_dict(self.add_prefix(loss_dict, prefix="train"))
        return loss_dict

    def validation_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss_dict = self.loss_step(batch, res)
        self.log_dict(self.add_prefix(loss_dict, prefix="val"), sync_dist=True)
        return loss_dict

    def test_step(self, batch, *args, **kwargs):
        res = self(batch)
        loss_dict = self.loss_step(batch, res)
        self.log_dict(self.add_prefix(loss_dict, prefix="test"), sync_dist=True)
        return loss_dict
