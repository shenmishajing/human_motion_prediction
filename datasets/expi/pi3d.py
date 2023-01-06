"""
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
MIT license.
"""
# pi3d.py

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import *

# from IPython import embed


class ExPiDataset(Dataset):
    def __init__(
        self, data_root, input_n, output_n, protocol, test_split=None, split="train"
    ):

        self.data_root = data_root
        self.input_n = input_n
        self.output_n = output_n
        self.protocol = protocol
        self.test_split = test_split
        self.is_train = split == "train"
        self.split = 0 if self.is_train else 1
        self.skip_rate = 1
        self.p3d = {}
        self.data_idx = []

        if self.protocol == "pro3":  # unseen action split
            if self.is_train:  # train on acro2
                acts = [
                    "2/a-frame",
                    "2/around-the-back",
                    "2/coochie",
                    "2/frog-classic",
                    "2/noser",
                    "2/toss-out",
                    "2/cartwheel",
                    "1/a-frame",
                    "1/around-the-back",
                    "1/coochie",
                    "1/frog-classic",
                    "1/noser",
                    "1/toss-out",
                    "1/cartwheel",
                ]
                subfix = [
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 4, 5, 6],
                    [1, 2, 3, 4, 6],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                ]

            else:  # test on acro1
                acts = [
                    "2/crunch-toast",
                    "2/frog-kick",
                    "2/ninja-kick",
                    "1/back-flip",
                    "1/big-ben",
                    "1/chandelle",
                    "1/check-the-change",
                    "1/frog-turn",
                    "1/twisted-toss",
                ]
                subfix = [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 4, 5, 8],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                ]

                if (
                    self.test_split is not None
                ):  # test per action for unseen action split
                    acts, subfix = [acts[self.test_split]], [subfix[self.test_split]]

        else:  # common action split and single action split
            if self.is_train:  # train on acro2
                acts = [
                    "2/a-frame",
                    "2/around-the-back",
                    "2/coochie",
                    "2/frog-classic",
                    "2/noser",
                    "2/toss-out",
                    "2/cartwheel",
                ]
                subfix = [
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [2, 3, 4, 5, 6],
                ]

                if self.protocol in [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                ]:  # train per action for single action split
                    acts = [acts[int(self.protocol)]]
                    subfix = [subfix[int(self.protocol)]]

            else:  # test on acro1
                acts = [
                    "1/a-frame",
                    "1/around-the-back",
                    "1/coochie",
                    "1/frog-classic",
                    "1/noser",
                    "1/toss-out",
                    "1/cartwheel",
                ]
                subfix = [
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 3, 4, 5],
                    [1, 2, 4, 5, 6],
                    [1, 2, 3, 4, 6],
                    [1, 2, 3, 4, 5],
                    [3, 4, 5, 6, 7],
                ]

                if (
                    self.test_split is not None
                ):  # test per action for common action split
                    acts, subfix = [acts[self.test_split]], [subfix[self.test_split]]
                if self.protocol in [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                ]:  # test per action for single action split
                    acts, subfix = [acts[int(self.protocol)]], [
                        subfix[int(self.protocol)]
                    ]

        key = 0
        for action_idx in range(len(acts)):
            subj_action = acts[action_idx]
            subj, action = subj_action.split("/")
            for subact_i in range(len(subfix[action_idx])):
                subact = subfix[action_idx][subact_i]
                filename = "{0}/acro{1}/{2}{3}/mocap_cleaned.tsv".format(
                    self.data_root, subj, action, subact
                )
                the_sequence = readCSVasFloat(filename, with_key=True)
                num_frames = the_sequence.shape[0]
                the_sequence = normExPI_2p_by_frame(the_sequence)
                the_sequence = torch.from_numpy(the_sequence).float()

                if self.is_train:  # train
                    seq_len = self.input_n + self.output_n
                    valid_frames = np.arange(
                        0, num_frames - seq_len + 1, self.skip_rate
                    )
                else:  # test
                    seq_len = self.input_n + 30
                    valid_frames = find_indices_64(num_frames, seq_len)

                p3d = the_sequence
                self.p3d[key] = p3d.view(num_frames, -1).data.numpy()
                tmp_data_idx_1 = [key] * len(valid_frames)
                tmp_data_idx_2 = list(valid_frames)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                key += 1

        self.dimension_use = np.arange(18 * 2 * 3)
        self.in_features = len(self.dimension_use)

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.input_n + self.output_n)
        data = self.p3d[key][fs][:, self.dimension_use]
        return data
