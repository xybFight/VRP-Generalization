from dataclasses import dataclass
import torch

from CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold, get_random_multi_dis_problems
import os
import pickle
import random
import numpy as np


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.num_dis = env_params['num_dis']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

        # AMDKD的数据
        # if 'load_path' in self.env_params:
        #     self.saved_depot_xy, self.saved_node_xy, self.saved_node_demand = to_tensor_for_CVRP(load_dataset(self.env_params['load_path']))
        #     self.saved_index = 0
        self.depot_node_xy_init = None
        if 'load_path' in self.env_params:
            if self.env_params['load_path'].endswith(".pkl"):
                self.FLAG__use_saved_problems = True
                self.saved_depot_xy, self.saved_node_xy, self.saved_node_demand = to_tensor_for_CVRP(load_dataset(self.env_params['load_path']))
                if 'val_load_path' in self.env_params:
                    self.saved_index_val = 0
                    self.saved_depot_xy_val, self.saved_node_xy_val, self.saved_node_demand_val = normalize_for_val(to_tensor_for_CVRP(load_dataset(self.env_params['val_load_path'])))
            else:
                self.all_depot_xy_origin, self.all_node_xy_origin, self.all_depot_xy, self.all_node_xy, self.all_node_demand = load_CVRPLIB(self.env_params['load_path'])
            self.saved_index = 0
        
        




    def load_problems(self, batch_size, aug_factor=1, few_shot=False, zero_shot=False):
        self.batch_size = batch_size
        if few_shot or zero_shot:
            if zero_shot:
                index = random.sample(range(self.saved_depot_xy_val.size(0)), batch_size)
            else:
                index = range(self.saved_index_val, self.saved_index_val + batch_size)
                self.saved_index_val += batch_size
            depot_xy = self.saved_depot_xy_val[index].unsqueeze(1).repeat(self.num_dis, 1, 1)
            # depot_xy = self.saved_depot_xy[index].repeat(self.num_dis, 1, 1)
            node_xy = self.saved_node_xy_val[index].repeat(self.num_dis, 1, 1)
            node_demand = self.saved_node_demand_val[index].repeat(self.num_dis, 1)
            if aug_factor > 1:
                if aug_factor == 8:
                    self.batch_size = self.batch_size * 8
                    depot_xy = augment_xy_data_by_8_fold(depot_xy)
                    node_xy = augment_xy_data_by_8_fold(node_xy)
                    node_demand = node_demand.repeat(8, 1)
                else:
                    raise NotImplementedError
            self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
            # shape: (batch, problem+1, 2)
            depot_demand = torch.zeros(size=(self.batch_size * self.num_dis, 1))
            # shape: (batch, 1)
            self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
            self.batch_size = self.batch_size * self.num_dis
        else:
            if 'load_path' in self.env_params: # 测试
                if self.env_params['load_path'].endswith(".pkl"):
                    depot_xy = self.saved_depot_xy[self.saved_index:(self.saved_index + batch_size)].unsqueeze(1)
                    # depot_xy = self.saved_depot_xy[self.saved_index:(self.saved_index + batch_size)]
                    node_xy = self.saved_node_xy[self.saved_index:(self.saved_index + batch_size)]
                    node_demand = self.saved_node_demand[self.saved_index:(self.saved_index + batch_size)]
                    self.saved_index += batch_size
                else:
                    # origin
                    depot_xy_init = self.all_depot_xy_origin[self.saved_index].repeat(self.num_dis, 1, 1)
                    node_xy_init = self.all_node_xy_origin[self.saved_index].repeat(self.num_dis, 1, 1)
                    node_demand = self.all_node_demand[self.saved_index].repeat(self.num_dis, 1)

                    # norm 
                    depot_xy = self.all_depot_xy[self.saved_index].repeat(self.num_dis, 1, 1)
                    node_xy = self.all_node_xy[self.saved_index].repeat(self.num_dis, 1, 1)

                    self.problem_size = node_xy.size(1)
                    self.pomo_size = self.problem_size
                    self.saved_index += batch_size
                    self.batch_size = self.batch_size * self.num_dis

                    # 给norm的例子加上增强
                    depot_xy_init = augment_xy_data_by_8_fold(depot_xy_init)
                    node_xy_init = augment_xy_data_by_8_fold(node_xy_init)
                    self.depot_node_xy_init = torch.cat((depot_xy_init, node_xy_init), dim=1)


            else:
                depot_xy, node_xy, node_demand = get_random_multi_dis_problems(batch_size, self.problem_size)
            if aug_factor > 1:
                if aug_factor == 8:
                    self.batch_size = self.batch_size * 8
                    depot_xy = augment_xy_data_by_8_fold(depot_xy)
                    node_xy = augment_xy_data_by_8_fold(node_xy)
                    node_demand = node_demand.repeat(8, 1)
                else:
                    raise NotImplementedError

            self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
            # shape: (batch, problem+1, 2)
            depot_demand = torch.zeros(size=(self.batch_size, 1))
            # shape: (batch, 1)
            self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
            # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][
            ~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        if self.depot_node_xy_init is None:
            all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        else:
            all_xy = self.depot_node_xy_init[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances


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


# 加载数据
def to_tensor_for_CVRP(data):
    for i in range(len(data)):
        depot_xy = torch.FloatTensor(data[i][0]).unsqueeze(0) if i == 0 else torch.cat(
            (depot_xy, torch.FloatTensor(data[i][0]).unsqueeze(0)), dim=0)
        node_xy = torch.FloatTensor(data[i][1]).unsqueeze(0).cuda() if i == 0 else torch.cat(
            (node_xy, torch.FloatTensor(data[i][1]).unsqueeze(0).cuda()), dim=0)
        node_demand = torch.FloatTensor(data[i][2]).unsqueeze(0) if i == 0 else torch.cat(
            (node_demand, torch.FloatTensor(data[i][2]).unsqueeze(0)), dim=0)
    node_demand = node_demand.to(node_xy.device) / float(data[0][3])
    return depot_xy.to(node_xy.device), node_xy, node_demand

def normalize_for_val(data):
    depot_xy,  node_xy, node_demand = data
    depot_node = torch.cat((depot_xy.unsqueeze(1), node_xy), dim=1)

    min_p = torch.min(depot_node, dim=1, keepdim=True)[0]
    max_p = torch.max(depot_node, dim=1, keepdim=True)[0]
    depot_node_norm = (depot_node - min_p) / (max_p - min_p)

    depot_xy_norm = depot_node_norm[:, 0, :]
    node_xy_norm = depot_node_norm[:, 1:, :]
    return depot_xy_norm, node_xy_norm, node_demand



def load_CVRPLIB(filename):

    all_depot_xy_origin, all_node_xy_origin, all_depot_xy_norm, all_node_xy_norm, all_node_demand= [], [], [], [], []

    for path in sorted(os.listdir(filename)):
        print(path)
        file = open(os.path.join(filename, path), "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
                i = i + dimension
            elif line.startswith('DEMAND_SECTION'):
                demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
                i = i + dimension
            i += 1
        original_locations = torch.tensor(locations[:, 1:], dtype=torch.float).unsqueeze(0)
        depot_xy, node_xy = torch.Tensor(original_locations[:, :1, :]), torch.Tensor(original_locations[:, 1:, :])
        node_demand = torch.Tensor(demand[1:, 1:].reshape((1, -1))) / capacity  # [1, n]

        # Norm
        depot_node = torch.cat((depot_xy, node_xy), dim=1)
        min_p = torch.min(original_locations, dim=1, keepdim=True)[0]
        max_p = torch.max(original_locations, dim=1, keepdim=True)[0]
        depot_node_norm = (original_locations - min_p) / (max_p - min_p)
        depot_xy_norm = depot_node_norm[:, 0, :].unsqueeze(0)
        node_xy_norm = depot_node_norm[:, 1:, :]
        # loc_scaler = 1000
        # locations_norm = original_locations / loc_scaler 
        # depot_xy_norm, node_xy_norm = torch.Tensor(locations_norm[:, :1, :]), torch.Tensor(locations_norm[:, 1:, :])
        
        # merge
        all_depot_xy_origin.append(depot_xy)
        all_node_xy_origin.append(node_xy)
        all_depot_xy_norm.append(depot_xy_norm)
        all_node_xy_norm.append(node_xy_norm)
        all_node_demand.append(node_demand)
    return all_depot_xy_origin, all_node_xy_origin, all_depot_xy_norm, all_node_xy_norm, all_node_demand

        