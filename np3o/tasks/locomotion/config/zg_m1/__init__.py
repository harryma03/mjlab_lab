"""ZG-M1 velocity-training task registration."""

from mjlab.tasks.registry import register_mjlab_task

from np3o.algorithms.np3o.runner import MjlabNP3ORunner

from ...rl.rl_cfg import d1_np3o_runner_cfg
from .zg_m1_flat_cfg import zg_m1_flat_env_cfg, zg_m1_flat_play_env_cfg
from .zg_m1_rough_cfg import zg_m1_rough_env_cfg, zg_m1_rough_play_env_cfg


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-ZG-M1",
    env_cfg=zg_m1_flat_env_cfg(),
    play_env_cfg=zg_m1_flat_play_env_cfg(),
    rl_cfg=d1_np3o_runner_cfg(experiment_name="zg_m1_flat", max_iterations=10000),
    runner_cls=MjlabNP3ORunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-ZG-M1",
    env_cfg=zg_m1_rough_env_cfg(),
    play_env_cfg=zg_m1_rough_play_env_cfg(),
    rl_cfg=d1_np3o_runner_cfg(experiment_name="zg_m1_rough"),
    runner_cls=MjlabNP3ORunner,
)
