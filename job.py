import argparse
import os
import shutil

import torch

from gflownet.tasks.seh_frag import SEHFragTrainer

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--wandb', type=str)
parser.add_argument('--part', type=int, default=None)
parser.add_argument('--total_parts', type=int, default=None)

args = parser.parse_args()

hps = {
    'log_dir': f"{args.log_dir}",
    'overwrite_existing_exp': True,
    'qm9_h5_path': '/data/chem/qm9/qm9.h5',
    'num_data_loader_workers': 0,
# 
    'num_training_steps': 20_000,
    'global_batch_size': 64,
    'validate_every': 500,
    'lr_decay': 20000,
    'temperature_dist_params': 32.0,
    'temperature_sample_dist': 'constant',
    'sampling_tau': 0.99,
    'mp_pickle_messages': True,
    'offline_ratio': 0.0,

    'data_dir': args.data_dir,

    'wandb': args.wandb,
}

if args.part is not None and args.total_parts is not None:
    if args.total_parts > 1:
        hps.update({'part': args.part})
        hps.update({'total_parts': args.total_parts})
        hps.upate({'offline_ratio' : 0.25})

if os.path.exists(hps['log_dir']):
    if hps['overwrite_existing_exp'] and hps['log_dir'] != hps.get('data_dir', 1234):
        shutil.rmtree(hps['log_dir'])
    else:
        raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it. This error will also be raised if the data_dir is the same as the log_dir.")
os.makedirs(hps['log_dir'])

trial = SEHFragTrainer(hps, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
trial.verbose = True
trial.run()
