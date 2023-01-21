"""
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPL license.
"""
# data_utils.py
# func utils for data

import os

import numpy as np
import torch
import torch.nn.functional as F

# from IPython import embed

###########################################
## func for reading and saving data


def read_data(filename, with_key=True):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    data = []
    lines = open(filename).readlines()
    if with_key:  # skip first line
        title = lines[0]
        lines = lines[1:]
    else:
        title = None
    for line in lines:
        line = line.strip().split(",")
        if len(line) > 0:
            data.append([float(x) for x in line])

    return torch.Tensor(data), title.strip()


def save_data(data, version_name, title, root, file, original_name="original"):
    output_file = os.path.join(
        root.replace(original_name, version_name),
        file.replace("tsv", "csv"),
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(
        output_file,
        data.detach().cpu().numpy(),
        delimiter=",",
        header=title,
        fmt="%.16g",
    )


###########################################
## func utils for norm/unnorm


def normExPI_xoz(img, P0, P1, P2):
    # P0: orig
    # P0-P1: axis x
    # P0-P1-P2: olane xoz

    X0 = P0
    X1 = (P1 - P0) / np.linalg.norm((P1 - P0)) + P0  # x
    X2 = (P2 - P0) / np.linalg.norm((P2 - P0)) + P0
    X3 = np.cross(X2 - P0, X1 - P0) + P0  # y
    ### x2 determine z -> x2 determine plane xoz
    X2 = np.cross(X1 - P0, X3 - P0) + P0  # z

    X = np.concatenate(
        (np.array([X0, X1, X2, X3]).transpose(), np.array([[1, 1, 1, 1]])), axis=0
    )
    Q = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]).transpose()
    M = Q.dot(np.linalg.pinv(X))

    img_norm = img.copy()
    for i in range(len(img)):
        tmp = img[i]
        tmp = np.concatenate((tmp, np.array([1])), axis=0)
        img_norm[i] = M.dot(tmp)
    return img_norm


def normExPI_2p_by_frame(seq):
    nb, dim = seq.shape  # nb_frames, dim=108
    seq_norm = seq.copy()
    for i in range(nb):
        img = seq[i].reshape((-1, 3))  # 36
        P0 = (img[10] + img[11]) / 2
        P1 = img[11]
        P2 = img[3]
        img_norm = normExPI_xoz(img, P0, P1, P2)
        seq_norm[i] = img_norm.reshape(dim)
    return seq_norm


def get_base_vec_matrix(origin, x=None, z=None, x_certain=True):
    """transfer data to origin, x, z coordinate

    Args:
        origin (torch.Tensor): shape [..., 3]
        x (torch.Tensor): shape [..., 3]
        z (torch.Tensor): shape [..., 3]
        x_certain (bool, optional): whether x or z is certain default x. Defaults to True.

    Returns:
        torch.Tensor: transfered data shape [..., 3]
    """

    if x is None:
        x_vec = torch.zeros_like(origin)
        x_vec[..., 0] = 1
    else:
        x_vec = F.normalize(x - origin, dim=-1)

    if z is None:
        z_vec = torch.zeros_like(origin)
        z_vec[..., 2] = 1
    else:
        z_vec = F.normalize(z - origin, dim=-1)

    y_vec = z_vec.cross(x_vec, dim=-1)
    if x_certain:
        z_vec = x_vec.cross(y_vec, dim=-1)
    else:
        x_vec = y_vec.cross(z_vec, dim=-1)

    return torch.stack([x_vec, y_vec, z_vec], dim=-2)


def transfer_data(data, origin, *args, **kwargs):
    """transfer data to origin, x, z coordinate

    Args:
        data (torch.Tensor): shape [..., 3]
        origin (torch.Tensor): shape [..., 3]
        x (torch.Tensor): shape [..., 3]
        z (torch.Tensor): shape [..., 3]
        x_certain (bool, optional): whether x or z is certain default x. Defaults to True.

    Returns:
        torch.Tensor: transfered data shape [..., 3]
    """

    return (data - origin)[..., None, :].matmul(
        get_base_vec_matrix(origin, *args, **kwargs).pinverse()
    )[..., 0, :]


def transfer_data_use_four_dim(data, origin, *args, **kwargs):
    """transfer data to origin, x, z coordinate, use four dim

    Args:
        data (torch.Tensor): shape [..., 3]
        origin (torch.Tensor): shape [..., 3]
        x (torch.Tensor): shape [..., 3]
        z (torch.Tensor): shape [..., 3]
        x_certain (bool, optional): whether x or z is certain default x. Defaults to True.

    Returns:
        torch.Tensor: transfered data shape [..., 3]
    """

    M = get_base_vec_matrix(origin, *args, **kwargs) + origin[..., None, :]
    M = torch.stack([M, origin[..., None, :]], dim=-2)
    M = torch.cat([M, torch.ones_like(M[..., 0, None])], dim=-1).pinverse()
    pad_data = torch.cat([data, torch.ones_like(data[..., 0, None])], dim=-1)[
        ..., None, :
    ]

    return pad_data.matmul(M)[..., 0, :3]


def calculate_spherical_coordinate_for_points(data, *args, **kwargs):
    """transfer points to origin, x, z spherical coordinate

    Args:
        data (torch.Tensor): shape [..., 3]
        origin (torch.Tensor): shape [..., 3]
        x (torch.Tensor): shape [..., 3]
        z (torch.Tensor): shape [..., 3]
        x_certain (bool, optional): whether x or z is certain default x. Defaults to False.

    Returns:
        torch.Tensor: transfered points shape [..., 3]
    """
    if args or kwargs:
        data = transfer_data(data, *args, **kwargs)
    r = torch.linalg.norm(data, dim=-1)
    theta = torch.acos(data[..., 2] / r)
    phi = torch.atan2(data[..., 1], data[..., 0])
    return torch.stack([r, theta, phi], dim=-1)


def calculate_cartesian_coordinate_for_points(data, origin=None):
    """transfer points to origin cartesian coordinate

    Args:
        data (torch.Tensor): shape [..., 3] in spherical coordinate
        origin (torch.Tensor): shape [..., 3]

    Returns:
        torch.Tensor: transfered points shape [..., 3]
    """
    x = data[..., 0] * torch.sin(data[..., 1]) * torch.cos(data[..., 2])
    y = data[..., 0] * torch.sin(data[..., 1]) * torch.sin(data[..., 2])
    z = data[..., 0] * torch.cos(data[..., 1])
    data = torch.stack([x, y, z], dim=-1)
    if origin is not None:
        data = data + origin
    return data


person_points = {
    "dfs": [
        ## head
        # fhead - back
        [0, 3],
        # lhead - fhead
        [1, 0],
        # rhead - fhead
        [2, 0],
        ## larm
        # lshoulder - back
        [4, 3],
        # lelbow - lshoulder
        [6, 4],
        # lwrist - lelbow
        [8, 6],
        ## rarm
        # rshoulder - back
        [5, 3],
        # relbow - rshoulder
        [7, 5],
        # rwrist - relbow
        [9, 7],
        ## lleg
        # lknee - lhip
        [12, 10],
        # lheel - lknee
        [14, 12],
        # ltoes - lheel
        [16, 14],
        ## rleg
        # rknee - rhip
        [13, 11],
        # rheel - rknee
        [15, 13],
        # rtoes - rheel
        [17, 15],
    ],
    "bfs": [
        ## from back
        # fhead - back
        [0, 3],
        # lshoulder - back
        [4, 3],
        # rshoulder - back
        [5, 3],
        ## from lhip
        # lknee - lhip
        [12, 10],
        ## from rhip
        # rknee - rhip
        [13, 11],
        ## from fhead
        # lhead - fhead
        [1, 0],
        # rhead - fhead
        [2, 0],
        ## from lshoulder
        # lelbow - lshoulder
        [6, 4],
        ## from rshoulder
        # relbow - rshoulder
        [7, 5],
        ## from lknee
        # lheel - lknee
        [14, 12],
        ## from rknee
        # rheel - rknee
        [15, 13],
        ## from lelbow
        # lwrist - lelbow
        [8, 6],
        ## from relbow
        # rwrist - relbow
        [9, 7],
        ## from lheel
        # ltoes - lheel
        [16, 14],
        ## from rheel
        # rtoes - rheel
        [17, 15],
    ],
}


def calculate_spherical_coordinate_for_person(data, deep_first=True, *args, **kwargs):
    """transfer 18 points of a person to spherical coordinate

    Args:
        data (torch.Tensor): shape [..., 18, 3]
        deep_first (boolen): points order, default deep first

    Returns:
        torch.Tensor: transfered points shape [..., 18, 3]
    """

    def _calculate_spherical_coordinate_for_point_list(data, points):
        return [
            calculate_spherical_coordinate_for_points(
                *[data[..., p, None, :] if isinstance(p, int) else p for p in point_seq]
            )
            for point_seq in points
        ]

    if args or kwargs:
        data = transfer_data(data, *args, **kwargs)

    # origin
    origin = (data[..., 10, None, :] + data[..., 11, None, :]) / 2

    points = [
        # origin
        [origin],
        # back - origin
        [3, origin],
        # rhip - origin
        [11, origin],
    ]
    res = _calculate_spherical_coordinate_for_point_list(data, points)

    # transfer data to origin, x, z coordinate
    data = transfer_data(
        data, origin, data[..., 11, None, :], data[..., 3, None, :], x_certain=False
    )

    res.extend(
        _calculate_spherical_coordinate_for_point_list(
            data, person_points["dfs" if deep_first else "bfs"]
        )
    )

    return torch.cat(res, dim=-2)


def calculate_cartesian_coordinate_for_person(data, deep_first=True, origin=None):
    """transfer 18 points of a person to cartesian coordinate

    Args:
        data (torch.Tensor): shape [..., 18, 3]
        deep_first (boolen): points order, default deep first

    Returns:
        torch.Tensor: transfered points shape [..., 18, 3]
    """

    def _calculate_cartesian_coordinate_for_point_list(data, points):
        res = [None for _ in range(18)]
        for i, p in enumerate(points):
            res[p[0]] = calculate_cartesian_coordinate_for_points(
                data[..., i + 3, None, :], res[p[1]]
            )
        return torch.cat([r for r in res if r is not None], dim=-2)

    # origin
    origin = calculate_cartesian_coordinate_for_points(data[..., 0, None, :])
    back = calculate_cartesian_coordinate_for_points(data[..., 1, None, :], origin)
    rhip = calculate_cartesian_coordinate_for_points(data[..., 2, None, :], origin)
    lhip = 2 * origin - rhip

    base_vec_martix = get_base_vec_matrix(origin, rhip, back, x_certain=False)[
        ..., 0, :, :
    ]
    data = _calculate_cartesian_coordinate_for_point_list(
        data, person_points["dfs" if deep_first else "bfs"]
    )
    data = data.matmul(base_vec_martix)

    return torch.cat(
        [
            data[..., :3, :],
            back,
            data[..., 3:9, :],
            lhip,
            rhip,
            data[..., 9:, :],
        ],
        dim=-2,
    )


def calculate_spherical_coordinate(data, cascade=True, *args, **kwargs):
    """transfer 18 points of n person to spherical coordinate

    Args:
        data (torch.Tensor): shape [..., n * 18, 3]
        cascade (boolen): person order, default calculate origin by previous person's origin
        deep_first (boolen): points order use by for_person func, default deep first

    Returns:
        torch.Tensor: transfered points shape [..., n * 18, 3]
    """
    data = data.reshape(*data.shape[:-2], -1, 18, 3)
    if cascade:
        # origin
        origin = (data[..., :-1, 10, None, :] + data[..., :-1, 11, None, :]) / 2
        data = torch.cat(
            [data[..., 0, None, :, :], data[..., 1:, :, :] - origin], dim=-3
        )
    return calculate_spherical_coordinate_for_person(data, *args, **kwargs).reshape(
        *data.shape[:-2], -1, 3
    )


def calculate_cartesian_coordinate(data, cascade=True, *args, **kwargs):
    """transfer 18 points of n person to cartesian coordinate

    Args:
        data (torch.Tensor): shape [..., n * 18, 3]
        cascade (boolen): person order, default calculate origin by previous person's origin
        deep_first (boolen): points order use by for_person func, default deep first

    Returns:
        torch.Tensor: transfered points shape [..., n * 18, 3]
    """
    data = data.reshape(*data.shape[:-2], -1, 18, 3)
    data = calculate_cartesian_coordinate_for_person(data, *args, **kwargs)
    if cascade:
        origin = None
        for i in range(data.shape[-3]):
            if origin is not None:
                data[..., i, None, :, :] = data[..., i, None, :, :] + origin
            origin = (
                data[..., i, None, 10, None, :] + data[..., i, None, 11, None, :]
            ) / 2
    return data


###########################################
## func utils for finding test samples


def find_indices_64(num_frames, seq_len):
    T = num_frames - seq_len + 1
    all_list = range(0, T)
    res_list = list(range(0, T, (int(T / 64) + 1)))
    if len(res_list) < 64:
        res_list += [x for x in all_list if x not in res_list][: 64 - len(res_list)]
    return res_list
