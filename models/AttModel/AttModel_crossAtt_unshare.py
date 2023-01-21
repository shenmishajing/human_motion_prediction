"""
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPL license.
"""
# AttModel_crossAtt_unshare.py

import torch
from torch import nn

from models import LightningModule

from . import GCN, XIA
from .utils import *

# from IPython import embed


class AttModel(LightningModule):
    def __init__(
        self,
        in_features=108,
        kernel_size=10,
        d_model=256,
        num_stage=12,
        dct_n=20,
        input_n=50,
        output_n=30,
    ):
        super(AttModel, self).__init__()

        self.in_features = int(in_features / 2)
        self.d_model = d_model
        self.dct_n = dct_n
        self.input_n = input_n
        self.output_n = output_n
        self.kernel_size = kernel_size  # to compute K_i
        self.chunk_size = 2 * kernel_size  # to compute V_i
        self.nb_kpts = int(in_features / 3)
        self.dim_xia_v = input_n - self.chunk_size + 1

        self.convQ = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.in_features,
                        out_channels=d_model,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=6,
                        bias=False,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=5,
                        bias=False,
                    ),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.in_features,
                        out_channels=d_model,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=6,
                        bias=False,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=5,
                        bias=False,
                    ),
                    nn.ReLU(),
                ),
            ]
        )
        self.convK1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.in_features,
                        out_channels=d_model,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.in_features,
                        out_channels=d_model,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.ReLU(),
                ),
            ]
        )
        self.convK2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=6,
                        bias=False,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=5,
                        bias=False,
                    ),
                    nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=6,
                        bias=False,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=5,
                        bias=False,
                    ),
                    nn.ReLU(),
                ),
            ]
        )
        self.gcn = nn.ModuleList(
            [
                GCN.GCN(
                    input_feature=(dct_n) * 2,
                    hidden_feature=d_model,
                    p_dropout=0.3,
                    num_stage=num_stage,
                    node_n=self.in_features,
                ),
                GCN.GCN(
                    input_feature=(dct_n) * 2,
                    hidden_feature=d_model,
                    p_dropout=0.3,
                    num_stage=num_stage,
                    node_n=self.in_features,
                ),
            ]
        )

        # d_model = 256
        self.update_k = nn.ModuleList(
            [
                XIA.XIA(embed_dim=d_model, nb_h=8, dropout=0.1),
                XIA.XIA(embed_dim=d_model, nb_h=8, dropout=0.1),
            ]
        )
        # in_features = 54
        self.update_v = nn.ModuleList(
            [
                XIA.XIA(embed_dim=self.in_features, nb_h=6, dropout=0.1),
                XIA.XIA(embed_dim=self.in_features, nb_h=6, dropout=0.1),
            ]
        )

    def forward(self, data):
        data = [data[..., : self.in_features], data[..., self.in_features :]]

        dct_m, idct_m = get_dct_matrix(self.chunk_size, data[0])  # (20, 20)

        # (31, 10)
        idx_key = (
            torch.arange(self.kernel_size)[None, :]
            + torch.arange(self.dim_xia_v)[:, None]
        )
        # (31, 20)
        idx_val = (
            torch.arange(self.chunk_size)[None, :]
            + torch.arange(self.dim_xia_v)[:, None]
        )
        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * self.kernel_size

        query_res, key_res, value_res, out = [[] for _ in range(4)]

        for i in range(2):
            # k,q
            # (bs, 54, 40)
            key = data[i].mT[..., : -self.kernel_size]
            # (bs, 54, 10)
            query = data[i].mT[..., -self.kernel_size :]
            key = self.convK1[i](key / 1000.0)  # (bs, d_model, 40)
            query = self.convQ[i](query / 1000.0)  # (bs, d_model, 1)
            key = key[..., idx_key].transpose(1, 2)  # (bs, 31, d_model, 10)
            # (bs*31, d_model, 10)
            key = key.reshape(-1, *key.shape[-2:])
            # v
            # (bs, vn, vl, 54) -> (bs x vn, vl, 54)
            value = data[i][:, idx_val]
            value = value.reshape(-1, *value.shape[-2:])
            # (bs x vn, 54, dct_n)
            value = dct_m[None, : self.dct_n].matmul(value).mT

            query_res.append(query)
            key_res.append(key)
            value_res.append(value)

        for i in range(2):
            ## update k v
            # (bs*31, d_model, 10) : (batch_size, E, L)
            key = self.update_k[i](*key_res)
            key = self.convK2[i](key).squeeze()  # (bs*31, d_model)
            # (bs, d_model, 31)
            key = key.reshape(-1, self.dim_xia_v, *key.shape[1:]).permute(0, 2, 1)

            # (bs*31, 54, dct_n): (batch_size, E, L)
            value = self.update_v[i](*value_res)
            # (bs, 31, 54*dct_n)
            value = value.reshape(data[i].shape[0], self.dim_xia_v, -1)

            # (bs, 1, d_model) x (bs, d_model, 31)
            score = query_res[i].mT.matmul(key) + 1e-15
            # (bs, 1, 31)
            att = score / (score.sum(dim=-1, keepdim=True))
            # (bs, 1, 31) x (bs, 31, 54*dct_n) -> (bs, 54, dct_n)
            dct_att = att.matmul(value)[:, 0].reshape(data[i].shape[0], -1, self.dct_n)

            # gcn
            input_gcn = data[i][:, idx]
            dct_in = dct_m[None, : self.dct_n].matmul(input_gcn).mT
            dct_in = torch.cat([dct_in, dct_att], dim=-1)
            dct_out = self.gcn[i](dct_in)

            # idct
            out.append(
                idct_m[None, :, : self.dct_n].matmul(dct_out[..., : self.dct_n].mT)
            )
        return torch.cat(out, axis=2)

    def forward_test(self, data):
        itr_test = int(self.output_n / self.kernel_size) + 1
        pred = []
        for _ in range(itr_test):
            data_out = self(data)[:, self.kernel_size :]
            pred.append(data_out)
            data = torch.cat((data[:, self.kernel_size :], data_out), axis=1)
        return torch.cat(pred, axis=1)[
            :, : self.output_n
        ]  # batch_size, out_len, nb_joints * joint_size

    def _loss_step(self, pred, gt):
        diff = torch.norm(gt - pred, dim=1)
        loss_l = torch.mean(diff[..., :54])
        loss_f = torch.mean(diff[..., 54:])
        loss = loss_f + loss_l * pow(10, -self.trainer.current_epoch)

        return {"loss": loss, "loss_l": loss_l, "loss_f": loss_f}

    def metric_step(self, batch):
        gt = batch[:, self.input_n :]
        pred = self.forward_test(batch[:, : self.input_n])
        ### evaluate ###
        gt = self.trainer.datamodule.dataset.decode_points_func(gt)
        pred = self.trainer.datamodule.dataset.decode_points_func(pred)
        ## JME
        jme = torch.mean(torch.norm(gt - pred, dim=-1))

        ## AME
        gt = gt.reshape(*gt.shape[:-1], -1, 3)
        pred = rigid_align(pred.reshape(*pred.shape[:-1], -1, 3), gt)

        ame = torch.mean(torch.norm(gt - pred, dim=-1))
        return {"jme": jme, "ame": ame}

    def training_step(self, batch, *args, **kwargs):
        loss_dict = self.loss_step(
            self(batch[:, : self.input_n])[:, self.kernel_size :],
            batch[:, self.input_n :],
        )
        self.log_dict(self.add_prefix(loss_dict, prefix="train"))
        return loss_dict

    def validation_step(self, batch, *args, **kwargs):
        metric_dict = self.metric_step(batch)
        self.log_dict(self.add_prefix(metric_dict, prefix="val"), sync_dist=True)
        return metric_dict

    def test_step(self, batch, *args, **kwargs):
        metric_dict = self.metric_step(batch)
        self.log_dict(self.add_prefix(metric_dict, prefix="test"), sync_dist=True)
        return metric_dict
