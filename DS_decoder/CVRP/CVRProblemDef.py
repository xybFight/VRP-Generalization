
import torch
import numpy as np
import math


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def get_random_multi_dis_problems(batch_size, problem_size):
    sub_batch_size = batch_size // 3

    # uniform
    uni_depot_xy = torch.rand(size=(sub_batch_size, 1, 2))
    # shape: (batch, 1, 2)
    uni_node_xy = torch.rand(size=(sub_batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    # cluster
    n_cluster = 3
    center = np.array([list(np.random.rand(n_cluster * 2)) for _ in range(sub_batch_size)])
    center = 0.2 + (0.8 - 0.2) * center
    std = 0.07
    for j in range(sub_batch_size):
        mean_x, mean_y = center[j, ::2], center[j, 1::2]
        coords = torch.zeros(problem_size + 1, 2)
        for i in range(n_cluster):
            if i < n_cluster - 1:
                coords[int((problem_size + 1) / n_cluster) * i:int((problem_size + 1) / n_cluster) * (i + 1)] = torch.cat((torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_x[i], std), torch.FloatTensor(int((problem_size + 1) / n_cluster), 1).normal_(mean_y[i], std)), dim=1)
            elif i == n_cluster - 1:
                coords[int((problem_size + 1) / n_cluster) * i:] = torch.cat((torch.FloatTensor((problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(mean_x[i], std), torch.FloatTensor((problem_size + 1) - int((problem_size + 1) / n_cluster) * i, 1).normal_(mean_y[i], std)), dim=1)

        coords = torch.where(coords > 1, torch.ones_like(coords), coords)
        coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
        depot_idx = int(np.random.choice(range(coords.shape[0]), 1))
        clu_node_xy = coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0) if j == 0 else torch.cat((clu_node_xy, coords[torch.arange(coords.size(0)) != depot_idx].unsqueeze(0)), dim=0)
        clu_depot_xy = coords[depot_idx].unsqueeze(0).unsqueeze(0) if j == 0 else torch.cat((clu_depot_xy, coords[depot_idx].unsqueeze(0).unsqueeze(0)), dim=0)

    # mixed
    mix_depot_xy = torch.rand(size=(sub_batch_size, 1, 2))
    n_cluster_mix = 1
    center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(sub_batch_size)])
    center = 0.2 + (0.8 - 0.2) * center
    std = 0.07
    for j in range(sub_batch_size):
        mean_x, mean_y = center[j, ::2], center[j, 1::2]
        mutate_idx = np.random.choice(range(problem_size), int(problem_size / 2), replace=False)
        coords = torch.FloatTensor(problem_size, 2).uniform_(0, 1)
        for i in range(n_cluster_mix):
            if i < n_cluster_mix - 1:
                coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:int(problem_size / n_cluster_mix / 2) * (i + 1)]] = torch.cat((torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_x[i], std), torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_y[i], std)), dim=1)
            elif i == n_cluster_mix - 1:
                coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:]] = torch.cat((torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i, 1).normal_(mean_x[i], std), torch.FloatTensor(int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i, 1).normal_(mean_y[i], std)), dim=1)

        coords = torch.where(coords > 1, torch.ones_like(coords), coords).to(mix_depot_xy.device)
        coords = torch.where(coords < 0, torch.zeros_like(coords), coords).to(mix_depot_xy.device)
        mix_node_xy = coords.unsqueeze(0) if j == 0 else torch.cat((mix_node_xy, coords.unsqueeze(0)), dim=0)

    
    demand_scaler = math.ceil(30 + problem_size/5) if problem_size >= 20 else 20
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)

    # 进行归一化，由于仓库位置也在其中，所以需要将仓库位置也加入到所有节点中进行归一化
    depot_xy = torch.cat(([uni_depot_xy, clu_depot_xy, mix_depot_xy]), dim=0)
    node_xy = torch.cat(([uni_node_xy, clu_node_xy, mix_node_xy]), dim=0)
    depot_node = torch.cat((depot_xy, node_xy), dim=1)

    min_p, _ = torch.min(depot_node, dim=1, keepdim=True)
    max_p, _ = torch.max(depot_node, dim=1, keepdim=True)
    max_diff_values, _ = torch.max(max_p - min_p, dim=-1)

    depot_node_norm = (depot_node - min_p) / max_diff_values.unsqueeze(2)

    depot_xy_norm = depot_node_norm[:, 0, :].unsqueeze(1)
    node_xy_norm = depot_node_norm[:, 1:, :]

    return depot_xy_norm, node_xy_norm, node_demand








def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data