"""
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
MIT license.
"""
# pi3d.py

from torch.utils.data import Dataset

from .utils import *

# from IPython import embed


class ExPIDataset(Dataset):
    def __init__(
        self,
        data_root,
        data_version,
        input_n,
        output_n,
        protocol,
        test_split=None,
        split="train",
    ):

        self.data_root = data_root
        self.data_version = data_version
        self.input_n = input_n
        self.output_n = output_n
        self.protocol = protocol
        self.test_split = test_split
        self.all_origin = data_version.startswith("all")
        self.cascade = "cascade" in data_version
        self.deep_first = "dfs" in data_version
        self.is_train = split == "train"
        self.data = []
        self.data_idx = []

        if "spherical" in data_version:
            self.coordinate = "spherical"
            self.decode_points_func = (
                lambda data: calculate_cartesian_coordinate_for_points(
                    data.reshape(*data.shape[:-1], -1, 3)
                ).reshape(*data.shape)
            )
        elif "person" in data_version:
            self.coordinate = "person"
            self.decode_points_func = lambda data: calculate_cartesian_coordinate(
                data.reshape(*data.shape[:-1], -1, 3), self.cascade, self.deep_first
            ).reshape(*data.shape)
        else:
            if "old_norm" in data_version:
                self.coordinate = "old_norm"
            else:
                self.coordinate = "cartesian"
            self.decode_points_func = lambda x: x

        # unseen action split
        if self.protocol == "unseen":

            # train on acro2
            if self.is_train:
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

            # test on acro1
            else:
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

                # test per action for unseen action split
                if self.test_split is not None:
                    acts, subfix = [acts[self.test_split]], [subfix[self.test_split]]

        # common action split and single action split
        else:
            # train on acro2
            if self.is_train:
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

                # train per action for single action split
                if self.protocol != "common":
                    acts = [acts[self.protocol]]
                    subfix = [subfix[self.protocol]]

            # test on acro1
            else:
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

                # test per action for common action split
                if self.test_split is not None:
                    acts, subfix = [acts[self.test_split]], [subfix[self.test_split]]
                # test per action for single action split
                elif self.protocol != "common":
                    acts, subfix = [acts[self.protocol]], [subfix[self.protocol]]

        for action_idx in range(len(acts)):
            subj_action = acts[action_idx]
            subj, action = subj_action.split("/")
            for subact_i in range(len(subfix[action_idx])):
                subact = subfix[action_idx][subact_i]
                filename = f"{self.data_root}/{self.data_version}/acro{subj}/{action}{subact}/mocap_cleaned.csv"
                the_sequence, _ = read_data(filename, with_key=True)
                num_frames = the_sequence.shape[0]

                if self.is_train:  # train
                    seq_len = self.input_n + self.output_n
                    valid_frames = list(range(num_frames - seq_len + 1))
                else:  # test
                    seq_len = self.input_n + 30
                    valid_frames = list(find_indices_64(num_frames, seq_len))

                self.data.append(the_sequence.reshape(num_frames, -1))
                self.data_idx.extend(
                    zip([len(self.data) - 1] * len(valid_frames), valid_frames)
                )

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, item):
        idx, start_frame = self.data_idx[item]
        return self.data[idx][start_frame : start_frame + self.input_n + self.output_n]
