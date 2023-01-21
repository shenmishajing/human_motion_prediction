import os

from datasets.expi import *


def get_count(data, n, topk=None):
    if topk is None:
        topk = n-1
    data = data.reshape(*data.shape[:-2], -1)
    inds = data.topk(topk, largest=False)[1]
    count = {}
    for i in inds.flatten():
        count[i.item()] = count.get(i.item(), 0) + 1
    count = sorted([(k, v) for k, v in count.items()], key=lambda x: x[1], reverse=True)
    return [(k // n, k % n) for k, v in sorted(count[:topk], key=lambda x: x[0])]


def main():
    n = 19
    data_path = "data/ExPI"
    mean = []
    std = []
    for root, _, files in os.walk(os.path.join(data_path, "original")):
        for file in files:
            if not file.endswith(".tsv"):
                continue
            data, title = read_data(os.path.join(root, file))
            data = data.reshape(data.shape[0], 2, -1, 3)
            data = torch.cat(
                [data, (data[..., 10, None, :] + data[..., 11, None, :]) / 2], dim=-2
            )
            distance = torch.linalg.norm(
                data[..., None, :] - data[..., None, :, :], dim=-1
            )
            mean.append(distance.mean(dim=0))
            std.append(distance.std(dim=0))
    mean = torch.stack(mean, dim=1)
    std = torch.stack(std, dim=1)
    std += torch.tril(torch.full((n, n), torch.inf), diagonal=0)
    res = std / mean

    for topk in [17, 20,25]:
        print(f"topk={topk}")
        for data in [std, res]:
            print(get_count(data, n, topk))


if __name__ == "__main__":
    main()
