import datetime
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from env.pointMass import PointMass
from ruamel.yaml import RoundTripDumper, dump

import wandb
from common.helper import Visualizer
from common.replay_buffer import (
    MyMultiStepMemory,
    MyPrioritizedMemory,
)
import os
from common.feature import pm_feature, quadcopter_feature
from common.util import (
    check_obs,
    check_act,
    dump_cfg,
    np2ts,
    to_batch,
    ts2np,
)

warnings.simplefilter("once", UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

task_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
raisim_unity_Path = task_path + "/../raisim_model/raisimUnity/raisimUnity.x86_64"
rsc_path = task_path + "/../raisim_model"
log_path = task_path + "/../log/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


class AbstractAgent:
    def __init__(
        self,
        seed=0,
        torch_api=False,
    ):
        self.device = device

        torch.manual_seed(seed)

        torch.autograd.set_detect_anomaly(torch_api)  # detect NaN
        torch.autograd.profiler.profile(torch_api)
        torch.autograd.profiler.emit_nvtx(torch_api)

    def run(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError


class RaisimAgent(AbstractAgent):
    @classmethod
    def default_config(cls):
        env_cfg = dict(
            env_name="pointmass1d",
            num_envs=512,
            episode_max_step=500,
            total_episodes=int(50),
            random_robot_state=True,
            random_target_state=True,
            simulation_dt=0.01,
            control_dt=0.04,
            num_threads=10,
            reward={"success": {"coeff": 1}},
            success_threshold=[1, 1, 1, 1],  # pos[m], vel[m/s], ang[rad], angvel[rad/s]
            single_task=False,  # overwrite all task_weight to nav_w
            seed=123,
            log_interval=5,
            eval=True,
            eval_interval=10,
            eval_episodes=1,
            save_model=True,  # save model after evaluation
            render=True,  # render env
            record=True,  # dump config, record video only works if env is rendered
            log_path=log_path,  # config, video, model log path
            rsc_path=rsc_path,  # model resource path
            raisim_unity_path=raisim_unity_Path,  # raisim unity path
            torch_api=False,  # detect torch NAN and make profiler
        )
        agent_cfg = dict(
            gamma=0.99,
            updates_per_step=1,
            reward_scale=1.0,
        )
        buffer_cfg = dict(
            capacity=1000000,
            mini_batch_size=128,
            min_n_experience=1024,
            multi_step=1,
            n_env=env_cfg["num_envs"],
            prioritize_replay=False,
            alpha=0.6,
            beta=0.4,
            beta_annealing=1e-4,
        )
        return {"env_cfg": env_cfg, "agent_cfg": agent_cfg, "buffer_cfg": buffer_cfg}

    def __init__(self, cfg):
        self.env_cfg = cfg["env_cfg"]
        self.agent_cfg = cfg["agent_cfg"]
        self.buffer_cfg = cfg["buffer_cfg"]

        super().__init__(self.env_cfg["seed"], self.env_cfg["torch_api"])

        self.env_name = self.env_cfg["env_name"]
        self.success_threshold = self.env_cfg.pop("success_threshold", [1, 1, 1, 1])

        if self.env_cfg.get(
            "single_task", False
        ):  # TODO: move this to the configuration
            self.task_weight = {  # an adhoc implementation for pm single task testing
                "nav_w": (1, 0, 0, 0, 0),  # p, v, ang, angvel, success
                "hov_w": (1, 0, 0, 0, 0),
                "nav_w_eval": (1, 0, 0, 0, 0),
                "hov_w_eval": (1, 0, 0, 0, 0),
            }
        else:
            self.task_weight = {
                "nav_w": (0.9, 0.1, 0, 0, 0),  # p, v, ang, angvel, success
                "hov_w": (0.1, 0.9, 0, 0, 0),
                "nav_w_eval": (0, 0, 0, 0, 1),
                "hov_w_eval": (0, 0, 0, 0, 1),
            }
        if "quadcopter" in self.env_name:
            self.task_weight = {
                "nav_w": (1, 1, 1, 10, 100),  # p, v, ang, angvel, success
                "hov_w": (1, 5, 5, 5, 100),
                "nav_w_eval": (1, 1, 1, 10, 10000),
                "hov_w_eval": (1, 5, 5, 5, 10000),
            }

        self.env_spec = raisim_multitask_env(
            env_name=self.env_name,
            task_weight=self.task_weight,
            success_threshold=self.success_threshold,
            n_env=self.env_cfg["num_envs"],
        )

        self.env, w, feature = self.env_spec.get_env_w_feature()
        self.env_max_episodes = self.env_spec.env_max_episodes

        self.n_env = self.env_cfg["num_envs"]
        self.episode_max_step = self.env_cfg["episode_max_step"]
        self.render = self.env_cfg["render"]
        self.log_interval = self.env_cfg["log_interval"]
        self.total_episodes = int(self.env_cfg["total_episodes"])
        self.total_timesteps = self.n_env * self.episode_max_step * self.total_episodes

        self.eval = self.env_cfg["eval"]
        self.eval_interval = self.env_cfg["eval_interval"]
        self.eval_episodes = self.env_cfg["eval_episodes"]
        self.record = self.env_cfg["record"]
        self.save_model = self.env_cfg["save_model"]

        w_train, w_eval = w[0], w[1]  # [F]
        self.w_navi, self.w_hover = w_train[0], w_train[1]  # [F]
        self.w_eval_navi, self.w_eval_hover = w_eval[0], w_eval[1]  # [F]
        self.w_init = torch.tile(self.w_navi, (self.n_env, 1))  # [N, F]
        self.w_eval_init = torch.tile(self.w_eval_navi, (self.n_env, 1))  # [N, F]
        self.w = self.w_init.clone().type(torch.float32)  # [N, F]
        self.w_eval = self.w_eval_init.clone().type(torch.float32)  # [N, F]

        self.feature = feature
        self.observation_dim = self.env.num_obs
        self.feature_dim = self.feature.dim
        self.action_dim = self.env.num_act
        self.observation_shape = [self.observation_dim]
        self.feature_shape = [self.feature_dim]
        self.action_shape = [self.action_dim]

        self.per = self.buffer_cfg["prioritize_replay"]
        memory = MyPrioritizedMemory if self.per else MyMultiStepMemory
        self.replay_buffer = memory(
            state_shape=self.observation_shape,
            feature_shape=self.feature_shape,
            action_shape=self.action_shape,
            device=device,
            **self.buffer_cfg,
        )
        self.mini_batch_size = int(self.buffer_cfg["mini_batch_size"])
        self.min_n_experience = int(self.buffer_cfg["min_n_experience"])

        self.gamma = int(self.agent_cfg["gamma"])
        self.updates_per_step = int(self.agent_cfg["updates_per_step"])
        self.reward_scale = int(self.agent_cfg["reward_scale"])

        log_dir = self.agent_cfg["name"] + "/" + "pointmass" + "/" + exp_date + "/"
        self.log_path = self.env_cfg["log_path"] + log_dir
        if self.record:
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            dump_cfg(self.log_path + "cfg", cfg)

        self.visualizer = Visualizer(
            self.env,
            raisim_unity_path=self.env_cfg["raisim_unity_path"],
            render=self.render,
            record=self.record,
            save_video_path=self.log_path,
        )

        self.steps = 0
        self.episodes = 0

        self.games_to_track = 100
        self.game_rewards = AverageMeter(1, self.games_to_track).to("cuda:0")
        self.game_lengths = AverageMeter(1, self.games_to_track).to("cuda:0")

    def run(self):
        # self.visualizer.spawn()

        while True:
            self.train_episode()
            if self.steps > self.total_timesteps:
                break

        # self.visualizer.kill()

    def train_episode(self):
        self.episodes += 1
        episode_r = episode_steps = 0
        done = False

        print("episode = ", self.episodes)

        s = self.reset_env()
        for _ in range(self.episode_max_step):
            a = self.act(s)
            # _, done = self.env.step(a)

            self.env.step(a)
            done = self.env.reset_buf.clone()

            # episodeRet = self.env.return_buf.clone()
            episodeLen = self.env.progress_buf.clone()

            # s_next = self.env.observe(update_statistics=False)
            s_next = self.env.obs_buf.clone()
            self.env.reset()

            r = self.calc_reward(s_next, self.w)
            masked_done = False if episode_steps >= self.episode_max_step else done
            self.save_to_buffer(s, a, r, s_next, done, masked_done)

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            s = s_next
            self.update_w(s, self.w, self.w_navi, self.w_hover)

            self.steps += self.n_env
            episode_steps += 1
            episode_r += r

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.size()[0]:
                self.game_rewards.update(episode_r[done_ids])
                self.game_lengths.update(episodeLen[done_ids])

            if episode_steps >= self.episode_max_step:
                break

        # if self.episodes % self.log_interval == 0:
        wandb.log({"reward/train": self.game_rewards.get_mean()})
        wandb.log({"length/train": self.game_lengths.get_mean()})

        if self.eval and (self.episodes % self.eval_interval == 0):
            self.evaluate()

    def update_w(self, s, w, w_navi, w_hover, thr=2):
        dist = torch.linalg.norm(s[:, 0:3], axis=1)
        w[torch.where(dist <= thr)[0], :] = w_hover
        w[torch.where(dist > thr)[0], :] = w_navi

    def is_update(self):
        return (
            len(self.replay_buffer) > self.mini_batch_size
            and self.steps >= self.min_n_experience
        )

    def reset_env(self):
        # s = self.env.reset()
        s = self.env.obs_buf.clone()
        if s is None:
            s = torch.zeros((self.n_env, self.env.num_obs))

        self.w = self.w_init.clone()
        self.w_eval = self.w_eval_init.clone()
        return s

    def save_to_buffer(self, s, a, r, s_next, done, masked_done):
        f = self.feature.extract(s)

        r = r[:, None] * self.reward_scale
        done = done[:, None]
        masked_done = masked_done[:, None]

        if self.per:
            error = self.calc_priority_error(
                to_batch(s, f, a, r, s_next, masked_done, device)
            )
            self.replay_buffer.append(s, f, a, r, s_next, masked_done, error, done)
        else:
            self.replay_buffer.append(s, f, a, r, s_next, masked_done, done)

    def evaluate(self):
        episodes = int(self.eval_episodes)
        if episodes == 0:
            return

        print(f"===== evaluate at episode: {self.episodes} ====")
        print(f"===== eval for running for {self.env_max_episodes} episodes ===")
        # self.visualizer.turn_on(self.episodes)

        returns = torch.zeros((episodes,), dtype=torch.float32)
        for i in range(episodes):
            episode_r = 0.0

            s = self.reset_env()
            for _ in range(self.env_max_episodes):
                a = self.act(s, "exploit")
                self.env.step(a)
                # s_next = self.env.observe(update_statistics=False)
                s_next = self.env.obs_buf.clone()
                self.env.reset()

                r = self.calc_reward(s_next, self.w_eval)

                s = s_next
                self.update_w(s, self.w_eval, self.w_eval_navi, self.w_eval_hover)
                episode_r += r

                if self.render:
                    time.sleep(0.04)

            returns[i] = torch.mean(episode_r).item()

        print(f"===== finish evaluate ====")
        # self.visualizer.turn_off()
        wandb.log({"reward/eval": torch.mean(returns).item()})

        if self.save_model:
            self.save_torch_model()

    def act(self, s, mode="explore"):
        if self.steps <= self.min_n_experience:
            a = 2 * torch.rand((self.n_env, self.env.num_act), device="cuda:0") - 1
        else:
            a = self.get_action(s, mode)

        a = check_act(a, self.action_dim)
        return a

    def get_action(self, s, mode):
        s, w = np2ts(s), np2ts(self.w)
        s = check_obs(s, self.observation_dim)

        with torch.no_grad():
            if mode == "explore":
                a = self.explore(s, w)
            elif mode == "exploit":
                a = self.exploit(s, w)
        return a

    def calc_reward(self, s, w):
        f = self.feature.extract(s)
        r = torch.sum(w * f, 1)
        return r

    def explore(self):
        raise NotImplementedError

    def exploit(self):
        raise NotImplementedError

    def calc_priority_error(self):
        raise NotImplementedError

    def save_torch_model(self):
        raise NotImplementedError

    def load_torch_model(self):
        raise NotImplementedError


class raisim_multitask_env:
    def __init__(
        self, env_name, task_weight, success_threshold=(1, 1, 1, 1), n_env=512
    ) -> None:
        self.env_name = env_name

        self.nav_w = task_weight["nav_w"]
        self.hov_w = task_weight["hov_w"]
        self.nav_w_eval = task_weight["nav_w_eval"]
        self.hov_w_eval = task_weight["hov_w_eval"]

        self.success_threshold = success_threshold

        self.n_env = n_env

        self.env_max_episodes = 500

    def get_env_w_feature(self):
        env = self.select_env(self.env_name)
        feature, combination = self.select_feature(
            self.env_name, self.success_threshold
        )
        task_w = self.define_tasks(self.env_name, combination)
        return env, task_w, feature

    def select_env(self, env_name):
        # if "pointmass1d" in env_name:
        #     from raisimGymTorch.env.bin import pointmass1

        #     env = quadcopter_task0
        if "isaacpointmass" in env_name:

            class pconfig:
                def __init__(self, n_env) -> None:
                    self.num_envs = n_env
                    self.sim_device = "cuda:0"
                    self.headless = True
                    self.compute_device_id = 0
                    self.graphics_device_id = 0

            args = pconfig(self.n_env)
            env = PointMass(args)
            self.env_max_episodes = env.max_episode_length
        else:
            raise NotImplementedError(
                "select one Raisim Env: pointmassXd, quadcopter_taskX"
            )

        return env

    def select_feature(self, env_name, success_threshold):
        if "pointmass" in env_name:
            # feature order: [pos, pos_norm, vel, vel_norm, ang, angvel, success]
            if "simple" in env_name:
                combination = [True, False, False, False, False, False, False]
            elif "augment" in env_name:
                combination = [True, True, True, True, False, False, True]
            else:
                combination = [True, False, False, True, False, False, True]
            feature = pm_feature(combination, success_threshold)
        elif "quadcopter" in env_name:
            combination = [True, True, True, True, True, True, True]
            feature = quadcopter_feature(combination, success_threshold)
        else:
            raise NotImplementedError("the feature is not implemented")

        return feature, combination

    def define_tasks(self, env_name, combination):
        def get_w(c, d, w):
            # feature order: [pos, pos_norm, vel, vel_norm, ang, angvel, success]
            w_pos = c[0] * (d * [w[0]] + (3 - d) * [0])
            w_pos_norm = c[1] * [w[0]]
            w_vel = c[2] * (d * [w[1]] + (3 - d) * [0])
            w_vel_norm = c[3] * [w[1]]
            w_ang = c[4] * (d * [w[2]] + (3 - d) * [0])
            w_angvel = c[5] * (d * [w[3]] + (3 - d) * [0])
            w_success = c[6] * [w[4]]
            return (
                w_pos + w_pos_norm + w_vel + w_vel_norm + w_ang + w_angvel + w_success
            )

        if "pointmass1d" in env_name:
            w_nav = get_w(combination, 1, self.nav_w)
            w_hov = get_w(combination, 1, self.hov_w)
            w_nav_eval = get_w(combination, 1, self.nav_w_eval)
            w_hov_eval = get_w(combination, 1, self.hov_w_eval)
        elif "pointmass2d" in env_name:
            w_nav = get_w(combination, 2, self.nav_w)
            w_hov = get_w(combination, 2, self.hov_w)
            w_nav_eval = get_w(combination, 2, self.nav_w_eval)
            w_hov_eval = get_w(combination, 2, self.hov_w_eval)
        elif "pointmass3d" in env_name:
            w_nav = get_w(combination, 3, self.nav_w)
            w_hov = get_w(combination, 3, self.hov_w)
            w_nav_eval = get_w(combination, 3, self.nav_w_eval)
            w_hov_eval = get_w(combination, 3, self.hov_w_eval)
        elif "isaacpointmass" in env_name:
            w_nav = get_w(combination, 3, self.nav_w)
            w_hov = get_w(combination, 3, self.hov_w)
            w_nav_eval = get_w(combination, 3, self.nav_w_eval)
            w_hov_eval = get_w(combination, 3, self.hov_w_eval)

        if "quadcopter" in env_name:
            w_nav = get_w(combination, 3, self.nav_w)
            w_hov = get_w(combination, 3, self.hov_w)
            w_nav_eval = get_w(combination, 3, self.nav_w_eval)
            w_hov_eval = get_w(combination, 3, self.hov_w_eval)

        tasks_train = (
            torch.tensor(w_nav, device="cuda:0"),
            torch.tensor(w_hov, device="cuda:0"),
        )
        tasks_eval = (
            torch.tensor(w_nav_eval, device="cuda:0"),
            torch.tensor(w_hov_eval, device="cuda:0"),
        )
        return (tasks_train, tasks_eval)
