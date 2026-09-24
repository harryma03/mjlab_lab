"""TITATIT velocity-tracking task registration for MjLab."""

from mjlab.tasks.registry import register_mjlab_task

from np3o.algorithms.np3o.runner import MjlabNP3ORunner

from .titatit_flat_cfg import (
    titatit_flat_env_cfg,
    titatit_flat_play_env_cfg,
)
from .titatit_rough_cfg import titatit_rough_env_cfg, titatit_rough_play_env_cfg

from ...rl.rl_cfg import d1_np3o_runner_cfg


_TITATIT_RL_CFG = d1_np3o_runner_cfg(
    experiment_name="titatit_flat",
    max_iterations=6000,
)
_TITATIT_RL_CFG["algorithm"]["entropy_coef"] = 0.003
_TITATIT_RL_CFG["policy"]["init_noise_std"] = 0.8

_TITATIT_ROUGH_RL_CFG = d1_np3o_runner_cfg(
    experiment_name="titatit_rough",
    max_iterations=15000,
)
_TITATIT_ROUGH_RL_CFG["algorithm"]["entropy_coef"] = 0.003
_TITATIT_ROUGH_RL_CFG["policy"]["init_noise_std"] = 0.8


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-TITATIT",

    env_cfg=titatit_flat_env_cfg(
        play=False,
    ),

    play_env_cfg=titatit_flat_play_env_cfg(),

    rl_cfg=_TITATIT_RL_CFG,

    runner_cls=MjlabNP3ORunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-TITATIT",
    env_cfg=titatit_rough_env_cfg(),
    play_env_cfg=titatit_rough_play_env_cfg(),
    rl_cfg=_TITATIT_ROUGH_RL_CFG,
    runner_cls=MjlabNP3ORunner,
)
