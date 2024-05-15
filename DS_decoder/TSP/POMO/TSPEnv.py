import random
from dataclasses import dataclass
import torch

from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold, get_random_multi_dis_problems
import os
import pickle
import numpy as np


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.num_dis = env_params['num_dis']
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        self.init_problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        if 'load_path' in self.env_params:
            if self.env_params['load_path'].endswith(".pkl"):
                self.saved_index = 0
                self.all_problems = torch.Tensor(load_dataset(self.env_params['load_path'])).squeeze(1)
                if 'val_load_path' in self.env_params:
                    self.saved_index_val = 0
                    self.all_problems_val = torch.Tensor(load_dataset(self.env_params['val_load_path'])).squeeze(1)
            else:
                # TSPLIB
                self.init_all_problems, self.all_problems = load_TSPLIB(self.env_params['load_path'])
                self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1, few_shot=False, zero_shot=False):
        self.batch_size = batch_size
        if few_shot or zero_shot:
            if zero_shot:
                index = random.sample(range(self.all_problems_val.size(0)), batch_size)
                self.problems = self.all_problems_val[index].repeat(self.num_dis, 1, 1)
            else:
                self.problems = self.all_problems_val[self.saved_index_val:(self.saved_index_val + batch_size)].repeat(self.num_dis, 1, 1)
                self.saved_index_val += batch_size
            self.batch_size = self.batch_size * self.num_dis
        else:
            if 'load_path' in self.env_params:
                if self.env_params['load_path'].endswith(".pkl"):
                    self.problems = self.all_problems[self.saved_index:(self.saved_index + batch_size)]
                    self.saved_index += batch_size
                else:
                    self.problems = self.all_problems[self.saved_index].repeat(self.num_dis, 1, 1)
                    self.init_problems = self.init_all_problems[self.saved_index].repeat(self.num_dis, 1, 1)
                    self.problem_size = self.problems.size(1)
                    self.pomo_size = self.problem_size
                    self.saved_index += batch_size
                    self.batch_size = self.batch_size * self.num_dis
            else:
                self.problems = get_random_multi_dis_problems(batch_size, self.problem_size)
            # problems.shape: (batch, problem, 2)
            if aug_factor > 1:
                if aug_factor == 8:
                    self.batch_size = self.batch_size * 8
                    self.problems = augment_xy_data_by_8_fold(self.problems)
                    if self.init_problems is not None:
                        self.init_problems = augment_xy_data_by_8_fold(self.init_problems)
                    # shape: (8*batch, problem, 2)
                else:
                    raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        if self.init_problems is None:
            seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        else:
            seq_expanded = self.init_problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


def data_norm(data):
    min_p = torch.min(data, dim=1, keepdim=True)[0]
    max_p = torch.max(data, dim=1, keepdim=True)[0]
    data_with_norm = (data - min_p) / (max_p - min_p)
    return data_with_norm


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def load_dataset(filename, disable_print=False):
    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
        if not disable_print:
            print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
        return data

def load_TSPLIB(filename):
    all_tsplib_data_origin, all_tsplib_data_norm = [], []

    for path in sorted(os.listdir(filename)):
        print(path)
        file = open(os.path.join(filename, path), "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
                i = i + dimension
            i += 1
        original_locations = torch.tensor(locations[:, 1:], dtype=torch.float).unsqueeze(0)

        problem_max_min = [torch.max(original_locations),torch.min(original_locations)]
        norm_locations = (original_locations - problem_max_min[1]) / (problem_max_min[0] - problem_max_min[1])
        # min_p = torch.min(original_locations, dim=1, keepdim=True)[0]
        # max_p = torch.max(original_locations, dim=1, keepdim=True)[0]
        # norm_locations =  (original_locations - min_p) / (max_p - min_p)
        all_tsplib_data_origin.append(original_locations)
        all_tsplib_data_norm.append(norm_locations)

    return all_tsplib_data_origin, all_tsplib_data_norm