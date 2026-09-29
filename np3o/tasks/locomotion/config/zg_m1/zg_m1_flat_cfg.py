"""ZG-M1 flat velocity task, using the D1 flat task as its baseline."""

from ..d1.d1_flat_cfg import d1_flat_env_cfg
from .zg_m1_cfg import configure_zg_m1


def zg_m1_flat_env_cfg(num_envs: int = 4096, play: bool = False, flatten_policy_history: bool = False):
    return configure_zg_m1(d1_flat_env_cfg(num_envs, play, flatten_policy_history))


def zg_m1_flat_play_env_cfg(num_envs: int = 50, flatten_policy_history: bool = False):
    return zg_m1_flat_env_cfg(num_envs, play=True, flatten_policy_history=flatten_policy_history)
