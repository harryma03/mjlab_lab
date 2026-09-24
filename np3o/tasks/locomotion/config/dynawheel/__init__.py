"""Dynawheel velocity-tracking task registration for mjlab."""

from mjlab.tasks.registry import register_mjlab_task

from np3o.algorithms.np3o.runner import MjlabNP3ORunner

from .dynawheel_flat_cfg import dynawheel_flat_env_cfg, dynawheel_flat_play_env_cfg
from .dynawheel_rough_cfg import dynawheel_rough_env_cfg, dynawheel_rough_play_env_cfg
from ...rl.rl_cfg import d1_np3o_runner_cfg


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Dynawheel",
    env_cfg=dynawheel_flat_env_cfg(play=False),
    play_env_cfg=dynawheel_flat_play_env_cfg(),
    rl_cfg=d1_np3o_runner_cfg(experiment_name="dynawheel_flat", max_iterations=10000),
    runner_cls=MjlabNP3ORunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Dynawheel",
    env_cfg=dynawheel_rough_env_cfg(play=False),
    play_env_cfg=dynawheel_rough_play_env_cfg(),
    rl_cfg=d1_np3o_runner_cfg(experiment_name="dynawheel_rough"),
    runner_cls=MjlabNP3ORunner,
)
