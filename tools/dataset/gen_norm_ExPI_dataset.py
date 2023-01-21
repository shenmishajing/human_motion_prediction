import os

from datasets.expi import *


def main():
    data_path = "data/ExPI"
    for root, _, files in os.walk(os.path.join(data_path, "original")):
        for file in files:
            if not file.endswith(".tsv"):
                continue
            data, title = read_data(os.path.join(root, file))

            save_data(
                torch.from_numpy(normExPI_2p_by_frame(data.detach().cpu().numpy())),
                "old_norm",
                title,
                root,
                file,
            )

            names = ["all_frame_origin", "first_frame_origin"]
            data = data.reshape(data.shape[0], -1, 3)
            datas = [
                torch.stack(
                    [
                        transfer_data(
                            data[i],
                            (data[i, 10, None] + data[i, 11, None]) / 2,
                            data[i, 11, None],
                            data[i, 3, None],
                        )
                        for i in range(data.shape[0])
                    ]
                ),
                transfer_data(
                    data,
                    (data[0, None, 10, None] + data[0, None, 11, None]) / 2,
                    data[0, None, 11, None],
                    data[0, None, 3, None],
                ),
            ]

            for cur_data, name in zip(datas, names):
                save_data(
                    cur_data.reshape(data.shape[0], -1),
                    name + "_cartesian",
                    title,
                    root,
                    file,
                )

                save_data(
                    calculate_spherical_coordinate_for_points(cur_data).reshape(
                        data.shape[0], -1
                    ),
                    name + "_spherical",
                    title,
                    root,
                    file,
                )

                for cascade in [True, False]:
                    for deep_first in [True, False]:
                        cur_name = (
                            name
                            + "_person"
                            + ("_cascade" if cascade else "")
                            + ("_dfs" if deep_first else "_bfs")
                        )

                        save_data(
                            calculate_spherical_coordinate(
                                cur_data, cascade, deep_first
                            ).reshape(data.shape[0], -1),
                            cur_name,
                            title,
                            root,
                            file,
                        )


if __name__ == "__main__":
    main()
