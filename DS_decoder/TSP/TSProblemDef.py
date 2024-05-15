import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems


def get_random_multi_dis_problems(batch_size, problem_size):
    sub_batch_size = batch_size // 3

    # uniform
    uni_problems = torch.rand(size=(sub_batch_size, problem_size, 2))

    # cluster

    clu_problems = get_cluster_problems(sub_batch_size, problem_size)

    # mix
    mix_problems = get_mixed_problems(sub_batch_size, problem_size)

    mix_problems = mix_problems.to(uni_problems.device)

    problems = torch.cat(([uni_problems, clu_problems, mix_problems]), dim=0)

    # 下面是归一化(对每个实例进行归一化,之前写错了)

    min_p, _ = torch.min(problems, dim=1, keepdim=True)
    max_p, _ = torch.max(problems, dim=1, keepdim=True)
    max_diff_values, _ = torch.max(max_p - min_p, dim=-1)
    problems_norm = (problems - min_p) / max_diff_values.unsqueeze(2)

    return problems_norm


def get_cluster_problems(batch_size, problem_size):
    n_cluster = 3
    center = np.array([list(np.random.rand(n_cluster * 2)) for _ in range(batch_size)])
    # center = distribution['lower'] + (distribution['upper'] - distribution['lower']) * center
    center = 0.2 + (0.8 - 0.2) * center
    std = 0.07
    for j in range(batch_size):
        mean_x, mean_y = center[j, ::2], center[j, 1::2]
        coords = torch.zeros(problem_size, 2)
        for i in range(n_cluster):
            if i < n_cluster - 1:
                coords[int((problem_size) / n_cluster) * i:int((problem_size) / n_cluster) * (i + 1)] = \
                    torch.cat((torch.FloatTensor(int((problem_size) / n_cluster), 1).normal_(mean_x[i], std),
                               torch.FloatTensor(int((problem_size) / n_cluster), 1).normal_(mean_y[i], std)), dim=1)
            elif i == n_cluster - 1:
                coords[int((problem_size) / n_cluster) * i:] = \
                    torch.cat((torch.FloatTensor((problem_size) - int((problem_size) / n_cluster) * i, 1).normal_(
                        mean_x[i], std),
                               torch.FloatTensor((problem_size) - int((problem_size) / n_cluster) * i, 1).normal_(
                                   mean_y[i], std)), dim=1)
        coords = torch.where(coords > 1, torch.ones_like(coords), coords)
        coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
        clu_problems = coords.unsqueeze(0) if j == 0 else torch.cat((clu_problems, coords.unsqueeze(0)), dim=0)
    return clu_problems


def get_mixed_problems(batch_size, problem_size):
    n_cluster_mix = 1
    center = np.array([list(np.random.rand(n_cluster_mix * 2)) for _ in range(batch_size)])
    center = 0.2 + (0.8 - 0.2) * center
    std = 0.07
    for j in range(batch_size):
        mean_x, mean_y = center[j, ::2], center[j, 1::2]
        mutate_idx = np.random.choice(range(problem_size), int(problem_size / 2), replace=False)
        coords = torch.FloatTensor(problem_size, 2).uniform_(0, 1)
        for i in range(n_cluster_mix):
            if i < n_cluster_mix - 1:
                coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:int(problem_size / n_cluster_mix / 2) * (
                        i + 1)]] = torch.cat((torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(
                    mean_x[i], std), torch.FloatTensor(int(problem_size / n_cluster_mix / 2), 1).normal_(mean_y[i],
                                                                                                         std)), dim=1)
            elif i == n_cluster_mix - 1:
                coords[mutate_idx[int(problem_size / n_cluster_mix / 2) * i:]] = torch.cat((torch.FloatTensor(
                    int(problem_size / 2) - int(problem_size / n_cluster_mix / 2) * i, 1).normal_(mean_x[i], std),
                                                                                            torch.FloatTensor(
                                                                                                int(problem_size / 2) - int(
                                                                                                    problem_size / n_cluster_mix / 2) * i,
                                                                                                1).normal_(mean_y[i],
                                                                                                           std)), dim=1)

        coords = torch.where(coords > 1, torch.ones_like(coords), coords)
        coords = torch.where(coords < 0, torch.zeros_like(coords), coords)
        mix_problems = coords.unsqueeze(0) if j == 0 else torch.cat((mix_problems, coords.unsqueeze(0)), dim=0)

    return mix_problems


def get_valid_muti_dis_problems(batch_size, problem_size):
    # 这里暂定是四种
    sub_batch_size = batch_size // 4

    # 第一种 diagonal分布
    dia_problems = torch.rand(size=(sub_batch_size, problem_size, 1)).repeat(1, 1, 2)
    rs = torch.rand(sub_batch_size).repeat_interleave(problem_size).view(sub_batch_size, problem_size)
    x1 = dia_problems[:, :, 1].mul(rs) + (1 - rs) / 2
    x2 = (1 - dia_problems[:, :, 1]).mul(rs) + (1 - rs) / 2
    ran_t = torch.randint(4, size=(sub_batch_size,))
    index1 = torch.where(ran_t == 0)[0]
    dia_problems[index1, :, 1] = x1[index1]
    index2 = torch.where(ran_t == 1)[0]
    dia_problems[index2, :, 1] = x2[index2]
    index3 = torch.where(ran_t == 2)[0]
    dia_problems[index3, :, 0] = x1[index3]
    index4 = torch.where(ran_t == 3)[0]
    dia_problems[index4, :, 0] = x2[index4]
    widths = (torch.rand((sub_batch_size)) * 0.15 + 0.05).repeat_interleave(problem_size).repeat_interleave(2).view(
        sub_batch_size, problem_size, 2)
    add_widths = torch.rand(sub_batch_size, problem_size, 2).mul(widths) - widths / 2
    dia_problems += add_widths

    # 第二种 uniform_rectangle
    uni_re_problems = torch.rand(size=(sub_batch_size, problem_size, 2))
    widths = torch.rand(sub_batch_size).repeat_interleave(problem_size).view(sub_batch_size, problem_size)
    uni_re_problems[:, :, 1] = uni_re_problems[:, :, 1] + (0.5 - widths / 2)
    ran_t = torch.randint(2, size=(sub_batch_size,))
    # 等于一的位置进行交换
    index = torch.where(ran_t == 1)[0]
    # 引入一个向量
    temp = uni_re_problems[index, :, 0]
    uni_re_problems[index, :, 0] = uni_re_problems[index, :, 1]
    uni_re_problems[index, :, 1] = temp

    # 再从OMNIVRP中找出两个


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems
