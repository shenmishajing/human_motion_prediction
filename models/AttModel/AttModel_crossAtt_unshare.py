"""
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPL license.
"""
# AttModel_crossAtt_unshare.py

import numpy as np
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

        self.convQ1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_features,
                out_channels=d_model,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=6, bias=False
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False
            ),
            nn.ReLU(),
        )
        self.convK11 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_features,
                out_channels=d_model,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
        )
        self.convK12 = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=6, bias=False
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False
            ),
            nn.ReLU(),
        )
        self.gcn1 = GCN.GCN(
            input_feature=(dct_n) * 2,
            hidden_feature=d_model,
            p_dropout=0.3,
            num_stage=num_stage,
            node_n=self.in_features,
        )

        self.convQ2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_features,
                out_channels=d_model,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=6, bias=False
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False
            ),
            nn.ReLU(),
        )
        self.convK21 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_features,
                out_channels=d_model,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
        )
        self.convK22 = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=6, bias=False
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False
            ),
            nn.ReLU(),
        )
        self.gcn2 = GCN.GCN(
            input_feature=(dct_n) * 2,
            hidden_feature=d_model,
            p_dropout=0.3,
            num_stage=num_stage,
            node_n=self.in_features,
        )

        self.update_k1 = XIA.XIA(
            embed_dim=d_model, nb_h=8, dropout=0.1
        )  # d_model = 256
        self.update_k2 = XIA.XIA(embed_dim=d_model, nb_h=8, dropout=0.1)
        self.update_v1 = XIA.XIA(
            embed_dim=self.in_features, nb_h=6, dropout=0.1
        )  # in_features = 54
        self.update_v2 = XIA.XIA(embed_dim=self.in_features, nb_h=6, dropout=0.1)

    def forward(self, src):
        src1 = src[:, :, : self.in_features]  # (bs, 50, 54)
        src2 = src[:, :, self.in_features :]  # (bs, 50, 54)
        src_ = [src1, src2]

        dct_m, idct_m = get_dct_matrix(self.chunk_size)  # (20, 20)
        dct_m = torch.as_tensor(dct_m, dtype=src1.dtype, device=src1.device)
        idct_m = torch.as_tensor(idct_m, dtype=src1.dtype, device=src1.device)
        dct_n = self.dct_n
        vn = self.dim_xia_v
        vl = self.chunk_size

        idx_val = np.expand_dims(np.arange(vl), axis=0) + np.expand_dims(
            np.arange(vn), axis=1
        )  # (31, 20)
        idx_key = np.expand_dims(np.arange(self.kernel_size), axis=0) + np.expand_dims(
            np.arange(vn), axis=1
        )  # (31, 10)
        src_tmp_, query_tmp_, key_tmp_, src_value_tmp_ = (
            [None] * 2,
            [None] * 2,
            [None] * 2,
            [None] * 2,
        )

        for i in range(2):
            src = src_[i]
            src_tmp = src.clone()
            bs = src.shape[0]

            # k,q
            src_key_tmp = src_tmp.transpose(1, 2)[
                :, :, : (self.input_n - self.kernel_size)
            ].clone()  # (bs, 54, 40)
            src_query_tmp = src_tmp.transpose(1, 2)[
                :, :, -self.kernel_size :
            ].clone()  # (bs, 54, 10)
            if i == 0:
                key_tmp = self.convK11(src_key_tmp / 1000.0)  # (bs, d_model, 40)
                query_tmp = self.convQ1(src_query_tmp / 1000.0)  # (bs, d_model, 1)
            else:
                key_tmp = self.convK21(src_key_tmp / 1000.0)
                query_tmp = self.convQ2(src_query_tmp / 1000.0)
            key_tmp = key_tmp[:, :, idx_key]  # (bs, d_model, 31, 10)
            key_tmp = key_tmp.transpose(1, 2).reshape(
                bs * vn, self.d_model, -1
            )  # (bs*31, d_model, 10)
            query_tmp = query_tmp.transpose(1, 2)  # (bs, 1, d_model)
            # v
            src_value_tmp = (
                src_tmp[:, idx_val].clone().reshape([bs * vn, vl, -1])
            )  # (bs, vn, vl, 54) -> (bs x vn, vl, 54)
            src_value_tmp = torch.matmul(
                dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp
            ).transpose(
                1, 2
            )  # (bs x vn, 54, dct_n)

            src_tmp_[i], query_tmp_[i], key_tmp_[i], src_value_tmp_[i] = (
                src_tmp,
                query_tmp,
                key_tmp,
                src_value_tmp,
            )

        ## update k v
        key_tmp_1 = self.update_k1(
            key_tmp_[0], key_tmp_[1]
        )  # (bs*31, d_model, 10) : (batch_size, E, L)
        key_tmp_1 = self.convK12(key_tmp_1).squeeze()  # (bs*31, d_model)
        key_tmp_1 = key_tmp_1.reshape(bs, vn, -1).permute(0, 2, 1)  # (bs, d_model, 31)
        src_value_tmp_1 = self.update_v1(
            src_value_tmp_[0], src_value_tmp_[1]
        )  # (bs*31, 54, dct_n): (batch_size, E, L)
        src_value_tmp_1 = src_value_tmp_1.reshape(bs, vn, -1)  # (bs, 31, 54*dct_n)

        key_tmp_2 = self.update_k2(key_tmp_[1], key_tmp_[0])
        key_tmp_2 = self.convK22(key_tmp_2).squeeze()
        key_tmp_2 = key_tmp_2.reshape(bs, vn, -1).permute(0, 2, 1)
        src_value_tmp_2 = self.update_v2(src_value_tmp_[1], src_value_tmp_[0])
        src_value_tmp_2 = src_value_tmp_2.reshape(bs, vn, -1)

        src_tmp_1, query_tmp_1, src_tmp_2, query_tmp_2 = (
            src_tmp_[0],
            query_tmp_[0],
            src_tmp_[1],
            query_tmp_[1],
        )

        score_tmp_1 = (
            torch.matmul(query_tmp_1, key_tmp_1) + 1e-15
        )  # (bs, 1, d_model) x (bs, d_model, 31)
        att_tmp_1 = score_tmp_1 / (
            torch.sum(score_tmp_1, dim=2, keepdim=True)
        )  # (bs, 1, 31)
        dct_att_tmp_1 = torch.matmul(att_tmp_1, src_value_tmp_1)[:, 0].reshape(
            [bs, -1, dct_n]
        )  # (bs, 54, 20)

        score_tmp_2 = torch.matmul(query_tmp_2, key_tmp_2) + 1e-15
        att_tmp_2 = score_tmp_2 / (torch.sum(score_tmp_2, dim=2, keepdim=True))
        dct_att_tmp_2 = torch.matmul(att_tmp_2, src_value_tmp_2)[:, 0].reshape(
            [bs, -1, dct_n]
        )

        # gcn
        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * self.kernel_size
        input_gcn_1 = src_tmp_1[:, idx]
        dct_in_tmp_1 = torch.matmul(
            dct_m[:dct_n].unsqueeze(dim=0), input_gcn_1
        ).transpose(1, 2)
        dct_in_tmp_1 = torch.cat([dct_in_tmp_1, dct_att_tmp_1], dim=-1)
        input_gcn_2 = src_tmp_2[:, idx]
        dct_in_tmp_2 = torch.matmul(
            dct_m[:dct_n].unsqueeze(dim=0), input_gcn_2
        ).transpose(1, 2)
        dct_in_tmp_2 = torch.cat([dct_in_tmp_2, dct_att_tmp_2], dim=-1)
        dct_out_tmp_1 = self.gcn1(dct_in_tmp_1)
        dct_out_tmp_2 = self.gcn2(dct_in_tmp_2)

        # idct
        out_gcn_1 = torch.matmul(
            idct_m[:, :dct_n].unsqueeze(dim=0),
            dct_out_tmp_1[:, :, :dct_n].transpose(1, 2),
        )
        out_gcn_2 = torch.matmul(
            idct_m[:, :dct_n].unsqueeze(dim=0),
            dct_out_tmp_2[:, :, :dct_n].transpose(1, 2),
        )
        return torch.cat([out_gcn_1, out_gcn_2], axis=2)

    def _loss_step(self, batch, res):
        loss_l = torch.mean(torch.norm((res - batch)[:, :, :54], dim=2))
        loss_f = torch.mean(torch.norm((res - batch)[:, :, 54:], dim=2))
        loss = loss_f + loss_l * pow(10, -self.trainer.current_epoch)

        return {"loss": loss, "loss_l": loss_l, "loss_f": loss_f}

    def metric_step(self, batch):
        batch = batch.reshape([-1, batch.shape[1], self.nb_kpts * 3])  # raw_in_n + 30
        data_in = batch[:, 50 - self.input_n : 50]  # in_n
        gt = batch[:, 50:].reshape([-1, self.output_n, self.nb_kpts, 3])  # out_n

        itr_test = int(self.output_n / self.kernel_size) + 1
        pred = []
        for _ in range(itr_test):
            data_out = self(data_in)
            data_out = data_out[:, self.kernel_size :]
            pred.append(data_out)
            data_in = torch.cat((data_in[:, self.kernel_size :], data_out), axis=1)
        pred = torch.cat(pred, axis=1)[:, : self.output_n].reshape(
            [-1, self.output_n, self.nb_kpts, 3]
        )  # batch_size, out_len, nb_joints, joint_size

        ### evaluate###
        ## JME
        joi = torch.mean(torch.norm(gt - pred, dim=3))

        ## AME
        pred_ali_l = rigid_align_torch(
            pred[:, :, : int(self.nb_kpts / 2)], gt[:, :, : int(self.nb_kpts / 2)]
        )
        pred_ali_f = rigid_align_torch(
            pred[:, :, int(self.nb_kpts / 2) :], gt[:, :, int(self.nb_kpts / 2) :]
        )
        pred_ali = torch.cat((pred_ali_l, pred_ali_f), axis=2)

        ali = torch.mean(torch.norm(gt - pred_ali, dim=3))
        return {"jme": joi, "ame": ali}

    def training_step(self, batch, *args, **kwargs):
        res = self(batch[:, : -self.kernel_size])
        loss_dict = self.loss_step(batch[:, -2 * self.kernel_size :], res)
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
