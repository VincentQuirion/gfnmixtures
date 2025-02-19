import ast
import copy
import os
import pickle
import shutil
import socket
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset
import wandb

from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.models import bengio2021flow, kmeans_classifier
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.train import FlatRewards, GFNTask, GFNTrainer, RewardScalar
from gflownet.utils.transforms import thermometer


class SEHTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching (TODO: port to this repo).
    """

    def __init__(
        self,
        dataset: Dataset,
        temperature_distribution: str,
        temperature_parameters: Tuple[float, float],
        num_thermometer_dim: int,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
        part: int = None,
        total_parts: int = None,
        data_dir: str = None,
    ):
        self.part = part
        self.total_parts = total_parts
        self.data_dir = data_dir

        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.num_thermometer_dim = num_thermometer_dim

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model)
        models_dict = {"seh": model}

        if self.part is not None:
            models_dict.update({"kmeans": 
                                kmeans_classifier.KMeansClassifier(f"{self.data_dir}/{self.total_parts}_kmeans_model.pkl")})

        return models_dict

    def sample_conditional_information(self, n: int) -> Dict[str, Tensor]:
        beta = None
        if self.temperature_sample_dist == "constant":
            assert type(self.temperature_dist_params) is float
            beta = np.array(self.temperature_dist_params).repeat(n).astype(np.float32)
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            if self.temperature_sample_dist == "gamma":
                loc, scale = self.temperature_dist_params
                beta = self.rng.gamma(loc, scale, n).astype(np.float32)
                upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
            elif self.temperature_sample_dist == "uniform":
                beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == "loguniform":
                low, high = np.log(self.temperature_dist_params)
                beta = np.exp(self.rng.uniform(low, high, n).astype(np.float32))
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == "beta":
                beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = 1
            beta_enc = thermometer(torch.tensor(beta), self.num_thermometer_dim, 0, upper_bound)

        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        return {"beta": beta, "encoding": beta_enc}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            flat_reward = torch.tensor(flat_reward)
        scalar_logreward = flat_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return RewardScalar(scalar_logreward * cond_info["beta"])

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu()
        preds[preds.isnan()] = 0
        
        if self.part is not None:
            parts = self.models['kmeans'](mols)

            if wandb.run is not None:
                # Count the occurrences of each number
                counts = np.bincount(parts, minlength=self.total_parts)
                good_part_frac = counts[self.part] / sum(counts)
                to_log = {'Specified part fraction': good_part_frac}
                if wandb.run.step == 1 or wandb.run.step % 250 == 0:
                    # Plot the histogram
                    plt.hist(range(self.total_parts), bins=self.total_parts, weights=counts, edgecolor='none')
                    plt.xlabel('Part')
                    plt.ylabel('Frequency')
                    plt.xticks(np.arange(0, self.total_parts, max(self.total_parts // 10, 1)))
                    # Set y-axis ticks to integers only
                    plt.yticks(np.arange(0, max(counts)+1, 5))
                    # Save the figure and log to wandb
                    fig = plt.gcf()
                    to_log.update({'Parts sampled from': wandb.Image(fig)})
                    plt.close(fig)
                wandb.log(to_log)
                
            preds[parts != self.part] = 0
        
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds), is_valid


class SEHFragTrainer(GFNTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            "hostname": socket.gethostname(),
            "bootstrap_own_reward": False,
            "learning_rate": 1e-4,
            "Z_learning_rate": 1e-4,
            "global_batch_size": 64,
            "num_emb": 128,
            "num_layers": 4,
            "tb_epsilon": None,
            "tb_p_b_is_parameterized": False,
            "illegal_action_logreward": -75,
            "reward_loss_multiplier": 1,
            "temperature_sample_dist": "uniform",
            "temperature_dist_params": (0.5, 32.0),
            "weight_decay": 1e-8,
            "num_data_loader_workers": 8,
            "momentum": 0.9,
            "adam_eps": 1e-8,
            "lr_decay": 20000,
            "Z_lr_decay": 20000,
            "clip_grad_type": "norm",
            "clip_grad_param": 10,
            "random_action_prob": 0.0,
            "valid_random_action_prob": 0.0,
            "sampling_tau": 0.0,
            "max_nodes": 9,
            "num_thermometer_dim": 32,
        }

    def setup_algo(self):
        self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, self.hps, max_nodes=self.hps["max_nodes"])

    def setup_task(self):
        self.task = SEHTask(
            dataset=self.training_data,
            temperature_distribution=self.hps["temperature_sample_dist"],
            temperature_parameters=self.hps["temperature_dist_params"],
            num_thermometer_dim=self.hps["num_thermometer_dim"],
            wrap_model=self._wrap_model_mp,
            part=self.hps.get("part"),
            total_parts=self.hps.get("total_parts"),
            data_dir=self.hps.get("data_dir"),
        )

    def setup_model(self):
        self.model = GraphTransformerGFN(self.ctx, num_emb=self.hps["num_emb"], num_layers=self.hps["num_layers"])

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.hps["max_nodes"], num_cond_dim=self.hps["num_thermometer_dim"]
        )

    def setup(self):
        hps = self.hps
        RDLogger.DisableLog("rdApp.*")
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.training_data = []
        self.test_data = []
        self.offline_ratio = self.hps.get("offline_ratio", 0)
        self.valid_offline_ratio = 0

        # Add training data if sampling from specified part
        if self.hps.get('part') is not None and self.offline_ratio is not None:
            data_dir = hps['data_dir']
            with open(f'{data_dir}/training_set_graphs.pkl', 'rb') as file:
                graphs = pickle.load(file)
            with open(f'{data_dir}/training_set_rewards.pkl', 'rb') as file:
                rewards = pickle.load(file)
            with open(f'{data_dir}/part_idxs.pkl', 'rb') as file:
                valid_idxs = pickle.load(file)
            self.training_data = TrainingDataset(valid_idxs[self.hps['part']], graphs, rewards)

        self.setup_env_context()
        self.setup_algo()
        self.setup_task()
        self.setup_model()

        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(self.model.logZ.parameters())
        non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        self.opt = torch.optim.Adam(
            non_Z_params,
            hps["learning_rate"],
            (hps["momentum"], 0.999),
            weight_decay=hps["weight_decay"],
            eps=hps["adam_eps"],
        )
        self.opt_Z = torch.optim.Adam(Z_params, hps["Z_learning_rate"], (0.9, 0.999))
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / hps["lr_decay"]))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(self.opt_Z, lambda steps: 2 ** (-steps / hps["Z_lr_decay"]))

        self.sampling_tau = hps["sampling_tau"]
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model
        eps = hps["tb_epsilon"]
        hps["tb_epsilon"] = ast.literal_eval(eps) if isinstance(eps, str) else eps

        self.mb_size = hps["global_batch_size"]
        self.clip_grad_param = hps["clip_grad_param"]
        self.clip_grad_callback = {
            "value": (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            "norm": (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            "none": (lambda x: None),
        }[hps["clip_grad_type"]]

    def step(self, loss: Tensor):
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        self.lr_sched.step()
        self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))


class TrainingDataset(Dataset):
    def __init__(self, in_part_idxs, graphs, rewards):
        self.in_part_idxs = in_part_idxs
        self.graphs = graphs
        self.rewards = rewards
    def __len__(self):
        return len(self.in_part_idxs)
    def __getitem__(self, idx):
        valid_idx = self.in_part_idxs[idx]
        return self.graphs[valid_idx], self.rewards[valid_idx]


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": "./logs/debug_run",
        "overwrite_existing_exp": True,
        "qm9_h5_path": "/data/chem/qm9/qm9.h5",
        "num_training_steps": 10_000,
        "validate_every": 1,
        "lr_decay": 20000,
        "sampling_tau": 0.99,
        "num_data_loader_workers": 8,
        "temperature_dist_params": (0.0, 64.0),
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHFragTrainer(hps, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    trial.verbose = True
    trial.run()


if __name__ == "__main__":
    main()
