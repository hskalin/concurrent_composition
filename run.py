import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
from common.util import omegaconf_to_dict, print_dict

from compose import CompositionAgent
from sac import SACAgent

import torch
import numpy as np

import wandb


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    # cfg_dict = omegaconf_to_dict(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(config=cfg_dict)

    cfg_dict = omegaconf_to_dict(wandb.config)

    cfg_dict["buffer"]["mini_batch_size"] *= int(cfg_dict["env"]["num_envs"] / 200)
    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    cfg_dict["env"]["episode_max_step"] = int(50 * (512 / cfg_dict["env"]["num_envs"]))

    print_dict(cfg_dict)

    # agent = CompositionAgent(cfg_dict)
    agent = SACAgent(cfg=cfg_dict)
    agent.run()
    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
