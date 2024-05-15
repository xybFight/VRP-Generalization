
import torch

import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from torch.optim import Adam as Optimizer

from utils.utils import *


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,
                 optimizer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.optimizer_params = optimizer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model_all = Model(False, **self.model_params)
        self.model = Model(True, **self.model_params)

        for p in self.model_all.encoder.parameters():
            p.requires_grad = False
        self.optimizer = Optimizer(self.model_all.decoder.parameters(), **self.optimizer_params['optimizer'])

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model_all.load_state_dict(checkpoint['model_state_dict'])
        self.model.encoder.load_state_dict(self.model_all.encoder.state_dict())

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        index = self.zero_few_shot()

        self.model.decoder.load_state_dict(self.model_all.decoder[index].state_dict())
        del self.model_all

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            print("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                print(" *** Test Done *** ")
                print(" NO-AUG SCORE: {:.8f} ".format(score_AM.avg))
                print(" AUGMENTATION SCORE: {:.8f} ".format(aug_score_AM.avg))

    def zero_few_shot(self):
        self.model_all.eval()
        with torch.no_grad():
            self.env.load_problems(self.tester_params['zero_shot_batch_size'], zero_shot=True)
            reset_state, _, _ = self.env.reset()
            self.model_all.pre_forward(reset_state)
            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model_all(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)

        index = reward.view(self.model_params['num_dis'], -1).mean(1).argmax()

        return index

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()
