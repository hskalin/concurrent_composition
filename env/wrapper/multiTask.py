from env import env_map
from common.feature import pm_feature
import torch


class MultiTaskEnv:
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.env = env_map[env_cfg["env_name"]](env_cfg)

        if env_cfg.get("single_task", False):
            task_weight = env_cfg["task"]["single"]
        else:
            task_weight = env_cfg["task"]["multi"]

        self.nav_w = task_weight["nav_w"]
        self.hov_w = task_weight["hov_w"]
        self.nav_w_eval = task_weight["nav_w_eval"]
        self.hov_w_eval = task_weight["hov_w_eval"]

        self.success_threshold = env_cfg.get("success_threshold", [1, 1, 1, 1])

    def define_tasks(self, env_cfg, combination):
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

    def getEnv(self):
        feature_type = self.env_cfg["feature"]["type"]
        combination = self.env_cfg["feature"][feature_type]
        task_w = self.define_tasks(self.env_cfg, combination)
        feature = pm_feature(combination, self.success_threshold)

        return self.env, task_w, feature
