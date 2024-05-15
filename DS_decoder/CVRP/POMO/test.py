##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
import random
import torch
import numpy as np


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src


from CVRPTester import CVRPTester as Tester


num_dis = 3
distribution = "mixed" # # uniform cluster mixed Implosion Grid Explosion Expansion
problem_size = 50

env_params = {
    'problem_size': problem_size,
    'pomo_size': problem_size,
    'load_path':'../../data/vrp/vrp_'+distribution+str(problem_size)+'_10000.pkl',
    'num_dis': num_dis,
    'val_load_path':'../../data/val/vrp_'+distribution+str(problem_size)+'_val_1000.pkl'
}



model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'num_dis': num_dis,

}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/CVRP_n'+str(problem_size)+'_checkpoint',  # directory path of pre-trained model and log files saved.
    },
    'test_episodes': 100*100,
    'test_batch_size': 100,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'zero_shot_batch_size': 50,
    'val_episodes': 1000
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']
logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}
def seed_everything(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()
    seed_everything(2024)
    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params,
                    optimizer_params=optimizer_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
