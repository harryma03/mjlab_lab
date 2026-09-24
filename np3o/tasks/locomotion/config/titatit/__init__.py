"""TITATIT velocity-tracking task registration for MjLab."""

from mjlab.tasks.registry import register_mjlab_task

from np3o.algorithms.np3o.runner import MjlabNP3ORunner

from .titatit_flat_cfg import (
    titatit_flat_env_cfg,
    titatit_flat_play_env_cfg,
)

from ...rl.rl_cfg import d1_np3o_runner_cfg


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-TITATIT",

    env_cfg=titatit_flat_env_cfg(
        play=False,
    ),

    play_env_cfg=titatit_flat_play_env_cfg(),

    rl_cfg=d1_np3o_runner_cfg(
        experiment_name="titatit_flat",
        max_iterations=6000,
    ),

    runner_cls=MjlabNP3ORunner,
)
