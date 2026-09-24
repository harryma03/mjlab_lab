# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1 flat task config.

Robot, terrain, observations, actions, events, rewards, terminations,
costs, and simulation all live here.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import torch

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.actuator.dc_actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
    base_ang_vel,
    base_lin_vel,
    flat_orientation_l2,
    joint_acc_l2,
    joint_pos_limits,
    joint_torques_l2,
    joint_vel_l2,
    joint_vel_rel,
    last_action,
    projected_gravity,
    reset_root_state_uniform,
    apply_external_force_torque,
    push_by_setting_velocity,
    time_out,
)
from mjlab.envs.mdp.dr import body_com_offset, geom_friction, pd_gains, pseudo_inertia
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    CurriculumTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch, RayCastSensorCfg, GridPatternCfg, ObjRef
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import (
    TerrainEntityCfg,
    TerrainGeneratorCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    HfPyramidSlopedTerrainCfg,
)
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.utils.noise.noise_cfg import UniformNoiseCfg

from ...cmdp.commands import UniformVelocityCommandCfg
from ...cmdp.curriculums import commands_vel_reward_based, terrain_levels_vel
from ...cmdp.observations import (
    contact_state,
    joint_kp_factor,
    joint_kd_factor,
    joint_pos_rel_without_wheel,
    velocity_commands_3d,
    safe_height_scan,
)
from ...cmdp.terminations import out_of_terrain_bounds, root_state_nonfinite
from ...cmdp.rewards import (
    action_rate_l2,
    ang_vel_xy_l2,
    base_height_l2,
    default_joint_l2,
    # foot_mirror,
    joint_pos_limits,
    joint_vel_l2,
    joint_vel_wheel_l2,
    lin_vel_z_l2,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
    undesired_contacts,
    upward,
)

# ===========================================================================
# 1. Robot
# ===========================================================================

_TITATIT_XML_PATH = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "assets"
    / "robots"
    / "titatit"
    / "mujoco"
    / "wheeled_titatit_mjlab.xml"
)


def _get_titatit_spec() -> mujoco.MjSpec:
    """Load the TITATIT MuJoCo model."""
    return mujoco.MjSpec.from_file(str(_TITATIT_XML_PATH))


# ---------------------------------------------------------------------------
# Actuators
#
# Parameters follow quadruped-wheel-titatit-rl:
#
# hip:
#   Kp = 25.0
#   Kd = 0.625
#   effort = 55 Nm
#   velocity = 52.4 rad/s
#
# thigh/calf:
#   Kp = 25.0
#   Kd = 0.625
#   effort = 55 Nm
#   velocity = 28.6 rad/s
#
# wheel:
#   velocity servo
#   damping = 0.5
#   effort = 10 Nm
# ---------------------------------------------------------------------------

_TITATIT_HIP_ACTUATOR = DcMotorActuatorCfg(
    target_names_expr=(".*_hip_joint",),
    stiffness=25.0,
    damping=0.625,
    effort_limit=55.0,
    saturation_effort=55.0,
    velocity_limit=52.4,
)

_TITATIT_THIGH_CALF_ACTUATOR = DcMotorActuatorCfg(
    target_names_expr=(".*_(thigh|calf)_joint",),
    stiffness=25.0,
    damping=0.625,
    effort_limit=55.0,
    saturation_effort=55.0,
    velocity_limit=28.6,
)

_TITATIT_WHEEL_ACTUATOR = IdealPdActuatorCfg(
    target_names_expr=(".*_foot_joint",),
    stiffness=0.0,
    damping=0.5,
    effort_limit=10.0,
)


_TITATIT_SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
_TITATIT_SOFT_VEL_LIMIT_FACTOR = 0.9
_TITATIT_SOFT_EFFORT_LIMIT_FACTOR = 0.9


# ---------------------------------------------------------------------------
# Joint / actuator patterns
# ---------------------------------------------------------------------------

LEG_JOINT_PATTERN = ".*_(hip|thigh|calf)_joint"
WHEEL_JOINT_PATTERN = ".*_foot_joint"
ALL_JOINT_PATTERN = ".*"

HIP_JOINT_PATTERN = ".*_hip_joint"
THIGH_CALF_JOINT_PATTERN = ".*_(thigh|calf)_joint"

LEG_ACTUATOR_PATTERN = ".*_(hip|thigh|calf)_joint"
WHEEL_ACTUATOR_PATTERN = ".*_foot_joint"
ALL_ACTUATOR_PATTERN = ".*"


# ---------------------------------------------------------------------------
# Cached SceneEntityCfg
# ---------------------------------------------------------------------------

_LEG_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=LEG_JOINT_PATTERN,
    preserve_order=True,
)

_WHEEL_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=WHEEL_JOINT_PATTERN,
    preserve_order=True,
)

_ALL_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=ALL_JOINT_PATTERN,
    preserve_order=True,
)

_LEG_ACTUATOR_CFG = SceneEntityCfg(
    "robot",
    actuator_names=LEG_ACTUATOR_PATTERN,
    preserve_order=True,
)

_WHEEL_ACTUATOR_CFG = SceneEntityCfg(
    "robot",
    actuator_names=WHEEL_ACTUATOR_PATTERN,
    preserve_order=True,
)

_ALL_ACTUATOR_CFG = SceneEntityCfg(
    "robot",
    actuator_names=ALL_ACTUATOR_PATTERN,
    preserve_order=True,
)


# ---------------------------------------------------------------------------
# Default standing pose
#
# Keep aligned with quadruped-wheel-titatit-rl.
# Rear TITA is mounted in the opposite longitudinal direction.
# ---------------------------------------------------------------------------

_TITATIT_DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,

    "FL_thigh_joint": 1.0,
    "FR_thigh_joint": 1.0,
    "RL_thigh_joint": -1.0,
    "RR_thigh_joint": -1.0,

    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": 1.5,
    "RR_calf_joint": 1.5,

    "FL_foot_joint": 0.0,
    "FR_foot_joint": 0.0,
    "RL_foot_joint": 0.0,
    "RR_foot_joint": 0.0,
}


# ---------------------------------------------------------------------------
# Joint limits from official TITATIT URDF
# ---------------------------------------------------------------------------

_TITATIT_JOINT_POS_LIMITS = {
    ".*_hip_joint": (
        -0.802851455917,
        0.802851455917,
    ),

    "F.*_thigh_joint": (
        -1.0471975512,
        4.18879020479,
    ),

    "R.*_thigh_joint": (
        -4.18879020479,
        1.0471975512,
    ),

    "F.*_calf_joint": (
        -2.69653369433,
        -0.916297857297,
    ),

    "RR_calf_joint": (
        -0.916297857297,
        2.69653369433,
    ),

    "RL_calf_joint": (
        0.916297857297,
        2.69653369433,
    ),

    ".*_foot_joint": (
        -1.0e6,
        1.0e6,
    ),
}


_TITATIT_JOINT_VEL_LIMITS = {
    ".*_hip_joint": 52.4,
    ".*_(thigh|calf)_joint": 28.6,
    ".*_foot_joint": 25.0,
}


_TITATIT_JOINT_EFFORT_LIMITS = {
    ".*_(hip|thigh|calf)_joint": 55.0,
    ".*_foot_joint": 10.0,
}


_TITATIT_ACTUATOR_EFFORT_LIMITS = {
    ".*_(hip|thigh|calf)_joint": 55.0,
    ".*_foot_joint": 10.0,
}


# ---------------------------------------------------------------------------
# Articulation
# ---------------------------------------------------------------------------

_TITATIT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        _TITATIT_HIP_ACTUATOR,
        _TITATIT_THIGH_CALF_ACTUATOR,
        _TITATIT_WHEEL_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=_TITATIT_SOFT_JOINT_POS_LIMIT_FACTOR,
)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

_TITATIT_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.44),
    joint_pos=_TITATIT_DEFAULT_JOINT_POS,
    joint_vel={".*": 0.0},
)


def get_titatit_cfg() -> EntityCfg:
    """Return a fresh TITATIT EntityCfg."""
    return EntityCfg(
        spec_fn=_get_titatit_spec,
        articulation=_TITATIT_ARTICULATION,
        init_state=_TITATIT_INIT_STATE,
    )

# ===========================================================================
# 2. Cost functions
# ===========================================================================

from ...cmdp.costs import (
    cost_default_joint,
    cost_dof_vel_limits,
    cost_hip_pos,
    cost_pos_limit,
    cost_torque_limit,
)


def _base_link_contact(
    env,
    sensor_name: str = "contact_forces",
    threshold: float = 1.0,
):
    """Terminate an episode when TITATIT base_link contacts terrain."""
    from mjlab.sensor import ContactSensor
    import torch

    sensor: ContactSensor = env.scene.sensors[sensor_name]

    cache = getattr(env, "_titatit_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_titatit_cache", cache)

    if "base_link_contact_ids" not in cache:
        base_ids = []
        num_slots = int(sensor.cfg.num_slots)

        for i, name in enumerate(sensor.primary_names):
            if "base_link" in name:
                start = i * num_slots
                base_ids.extend(
                    range(start, start + num_slots)
                )

        cache["base_link_contact_ids"] = torch.tensor(
            base_ids,
            device=env.device,
            dtype=torch.long,
        )

    body_ids = cache["base_link_contact_ids"]

    if len(body_ids) == 0:
        return torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )

    net_forces = sensor.data.force[:, body_ids, :]

    net_forces = torch.nan_to_num(
        net_forces,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return (
        net_forces.norm(dim=-1)
        .max(dim=-1)
        .values
        > threshold
    )


def _titatit_flat_cost_terms() -> dict:
    """Return cost terms for D1 NP3O constrained training.

    Five terms matching the IsaacLab ``D1FlatCfg.costs``:

    - ``pos_limit``        — joint position limit violations (all leg joints)
    - ``torque_limit``     — joint torque limit violations (all leg actuators)
    - ``dof_vel_limits``   — joint velocity limit violations
    - ``hip_pos``          — hip joint deviation from zero (L2)
    - ``default_joint``    — thigh + calf deviation from default pose (L1)
    """
    from np3o.algorithms.np3o.cost_manager import CostTermCfg

    return {
        "pos_limit": CostTermCfg(
            func=cost_pos_limit,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
                "joint_pos_limit_patterns": _TITATIT_JOINT_POS_LIMITS,
                "soft_ratio": _TITATIT_SOFT_JOINT_POS_LIMIT_FACTOR,
            },
        ),
        "torque_limit": CostTermCfg(
            func=cost_torque_limit,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=LEG_ACTUATOR_PATTERN),
                "actuator_effort_limit_patterns": _TITATIT_ACTUATOR_EFFORT_LIMITS,
                "soft_ratio": _TITATIT_SOFT_EFFORT_LIMIT_FACTOR,
            },
        ),
        "dof_vel_limits": CostTermCfg(
            func=cost_dof_vel_limits,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
                "joint_vel_limit_patterns": _TITATIT_JOINT_VEL_LIMITS,
                "soft_ratio": _TITATIT_SOFT_VEL_LIMIT_FACTOR,
            },
        ),
        "hip_pos": CostTermCfg(
            func=cost_hip_pos,
            scale=2.0, d_value=0.0, k_value=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_JOINT_PATTERN)},
        ),
        "default_joint": CostTermCfg(
            func=cost_default_joint,
            scale=0.2, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=THIGH_CALF_JOINT_PATTERN),
                "default_joint_pos_patterns": _TITATIT_DEFAULT_JOINT_POS,
            },
        ),
    }


# ---- reset helper: set PD position targets to default joint pose ----------


def _apply_default_joint_pos_target(
    env, env_ids, asset_cfg: SceneEntityCfg = _ALL_JOINT_CFG,
) -> None:
    """Set PD position targets to the default joint pose during reset.

    mjlab zeroes ``joint_pos_target`` inside ``sim.reset()``.  If this event does
    not run afterwards, the position-servo actuators see a target of 0 rad
    (all joints fully extended) and saturation-limit forces are produced for
    joints whose default pose is far from zero (calf ≈ -1.5, thigh ≈ 0.8).
    """
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    asset.set_joint_position_target(q_default, joint_ids=asset_cfg.joint_ids)


def _reset_joints_by_scale(
    env, env_ids, asset_cfg: SceneEntityCfg = _ALL_JOINT_CFG,
    position_scale_range: tuple[float, float] = (0.5, 1.5),
) -> None:
    """Reset joint positions by scaling the default pose.

    Matches IsaacGym's ``default_dof_pos * torch_rand_float(0.5, 1.5, ...)`` and
    DDT_Lab's ``reset_joints_by_scale``. Joint velocities are zeroed.
    """
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]

    default_joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids]
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]

    scale = torch.empty_like(default_joint_pos).uniform_(*position_scale_range)
    joint_pos = (default_joint_pos * scale).clamp_(
        soft_joint_pos_limits[..., 0], soft_joint_pos_limits[..., 1]
    )
    joint_vel = torch.zeros_like(joint_pos)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )


def _filter_zero_weight_rewards(
    rewards: dict[str, RewardTermCfg],
) -> dict[str, RewardTermCfg]:
    """Drop reward terms whose weight is exactly zero.

    Keeps the config aligned with the original scales (where several terms are
    set to 0.0) while preventing them from being printed/computed in mjlab.
    """
    return {name: term for name, term in rewards.items() if term.weight != 0.0}


# ===========================================================================
# 3. Environment constants
# ===========================================================================

# ---------------------------------------------------------------------------
# Flat/easy-heightfield terrain (owned by this file)
# ---------------------------------------------------------------------------

_FLAT_SLOPE_RANGE = (0.0, 0.2)
_FLAT_RANDOM_NOISE_RANGE = (0.0, 0.05)
_FLAT_WAVE_AMPLITUDE = (0.0, 0.2)
_FLAT_WAVE_COUNT = 2

_FLAT_PROP_RANDOM = 0.4
_FLAT_PROP_WAVES = 0.2
_FLAT_PROP_SLOPES = 0.4

_FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=10,
    border_width=10.0,   # 关键：给整个 terrain grid 外面加 20m 缓冲区
    curriculum=True,
    sub_terrains={
        "random_uniform": HfRandomUniformTerrainCfg(
            proportion=_FLAT_PROP_RANDOM,
            noise_range=_FLAT_RANDOM_NOISE_RANGE,

        ),
        "waves": HfWaveTerrainCfg(
            proportion=_FLAT_PROP_WAVES,
            amplitude_range=_FLAT_WAVE_AMPLITUDE,
            num_waves=_FLAT_WAVE_COUNT,

        ),
        "slopes": HfPyramidSlopedTerrainCfg(
            proportion=_FLAT_PROP_SLOPES,
            slope_range=_FLAT_SLOPE_RANGE,
            platform_width=2.0,
        ),
    },
)

# ===========================================================================
# 4. TITATIT flat environment
# ===========================================================================


def titatit_flat_env_cfg(
    num_envs: int = 4096,
    play: bool = False,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the TITATIT velocity-tracking environment configuration.

    Args:
        terrain_generator: Procedural terrain config. ``None`` for a flat plane,
            ``_FLAT_TERRAIN_CFG`` for heightfield flat.
        num_envs: Number of parallel environments.
        play: If True, disable noise/randomization and reduce env count.
        flatten_policy_history: If True (standard PPO), flatten the policy history dim.
            If False (NP3O), keep 3D shape for the BarlowTwins encoder.
    """
    terrain_generator = _FLAT_TERRAIN_CFG
    has_terrain = True

    # ---- Scene ----
    terrain_cfg = TerrainEntityCfg(
        terrain_type="generator" if has_terrain else "plane",
        terrain_generator=terrain_generator,
        max_init_terrain_level=5 if has_terrain else None,
        env_spacing=2.5,
    )

    sensors: list = [
        ContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            fields=("found", "force"),
            track_air_time=True,
            # Two frames are enough for contact-state/undesired-contact rewards.
            # Keeping four frames for every body in every env noticeably increases
            # GPU memory on both flat and rough tasks.
            history_length=2,
        ),
    ]

    if has_terrain:
        sensors.append(
            RayCastSensorCfg(
                name="height_scanner",
                frame=ObjRef(type="body", name="base_link", entity="robot"),
                pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
                ray_alignment="yaw",
                max_distance=5.0,
                exclude_parent_body=True,
            ),
        )

    scene = SceneCfg(
        terrain=terrain_cfg,
        entities={"robot": get_titatit_cfg()},
        sensors=tuple(sensors),
        num_envs=num_envs if not play else 50,
        env_spacing=2.5,
    )

    # ---- Observations ----
    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=base_ang_vel, params={"asset_cfg": _ALL_JOINT_CFG},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity, params={"asset_cfg": _ALL_JOINT_CFG},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0,
        ),
        "velocity_commands": ObservationTermCfg(
            func=velocity_commands_3d, params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0), scale=(2.0, 2.0, 0.25),
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel_without_wheel,
            params={
                "asset_cfg": _ALL_JOINT_CFG,
                "wheel_asset_cfg": _WHEEL_JOINT_CFG,
                "default_joint_pos_patterns": _TITATIT_DEFAULT_JOINT_POS,
            },
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01), clip=(-100.0, 100.0), scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel, params={"asset_cfg": _ALL_JOINT_CFG},
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5), clip=(-100.0, 100.0), scale=0.05,
        ),
        "actions": ObservationTermCfg(
            func=last_action, clip=(-100.0, 100.0), scale=1.0,
        ),
    }

    critic_terms = {
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel, clip=(-100.0, 100.0), scale=2.0),
        "base_ang_vel": ObservationTermCfg(
            func=base_ang_vel, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(-100.0, 100.0), scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=projected_gravity, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(-100.0, 100.0), scale=1.0,
        ),
        "velocity_commands": ObservationTermCfg(
            func=velocity_commands_3d, params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0), scale=(2.0, 2.0, 0.25),
        ),
        "joint_pos": ObservationTermCfg(
            func=joint_pos_rel_without_wheel,
            params={
                "asset_cfg": _ALL_JOINT_CFG,
                "wheel_asset_cfg": _WHEEL_JOINT_CFG,
                "default_joint_pos_patterns": _TITATIT_DEFAULT_JOINT_POS,
            },
            clip=(-100.0, 100.0), scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=joint_vel_rel, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(-100.0, 100.0), scale=0.05,
        ),
        "actions": ObservationTermCfg(func=last_action, clip=(-100.0, 100.0), scale=1.0),
    }

    priv_terms = {
        "contact_state": ObservationTermCfg(
            func=contact_state, params={"sensor_name": "contact_forces", "body_names": ".*_foot"},
            clip=(-1.0, 1.0), scale=1.0,
        ),
        "joint_kp_factor": ObservationTermCfg(
            func=joint_kp_factor, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(0.0, 2.0), scale=1.0,
        ),
        "joint_kd_factor": ObservationTermCfg(
            func=joint_kd_factor, params={"asset_cfg": _ALL_JOINT_CFG}, clip=(0.0, 2.0), scale=1.0,
        ),
    }

    scanner_terms = {}
    if has_terrain:
        scanner_terms["height_scan"] = ObservationTermCfg(
            func=safe_height_scan, params={"sensor_name": "height_scanner"},
            clip=(-1.0, 1.0), scale=1.0,
        )

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms, enable_corruption=not play,
            concatenate_terms=True, history_length=10, flatten_history_dim=flatten_policy_history,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, enable_corruption=False, concatenate_terms=True,
        ),
        "priv": ObservationGroupCfg(
            terms=priv_terms, enable_corruption=False, concatenate_terms=True,
        ),
    }
    if has_terrain:
        observations["scanner"] = ObservationGroupCfg(
            terms=scanner_terms, enable_corruption=False, concatenate_terms=True,
        )

    # ---- Action scales (mirrors D1FlatCfg.control) ----
    _action_scale = 0.25
    _hip_scale_reduction = 0.5

    # ---- Actions ----
    actions = {}
    for leg in ("FL", "FR", "RL", "RR"):
        actions[f"{leg.lower()}_leg_pos"] = JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint"),
            scale={
                ".*_hip_joint": _action_scale * _hip_scale_reduction,
                ".*_thigh_joint": _action_scale,
                ".*_calf_joint": _action_scale,
            },
            clip={".*": (-100.0, 100.0)},
            preserve_order=True,
        )
        actions[f"{leg.lower()}_foot_vel"] = JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=(f"{leg}_foot_joint",),
            scale=5.0,
            clip={".*": (-100.0, 100.0)},
            preserve_order=True,
        )

    # ---- Events ----
    events: dict[str, EventTermCfg] = {}

    if not play:
        events["randomize_actuator_gains"] = EventTermCfg(
            func=pd_gains, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
                "kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2),
                "operation": "scale", "distribution": "uniform",
            },
        )

        # Domain randomization: robot body mass+inertia, CoM offset, and surface friction.
        # pseudo_inertia scales mass and inertia together, matching IsaacLab's
        # randomize_rigid_body_mass with recompute_inertia=True.
        events["add_base_mass"] = EventTermCfg(
            func=pseudo_inertia, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "alpha_range": (-0.05, 0.08), "distribution": "uniform",
            },
        )
        events["add_base_com"] = EventTermCfg(
            func=body_com_offset, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "ranges": (-0.1, 0.1),
                "operation": "add", "distribution": "uniform",
            },
        )
        events["robot_geom_friction"] = EventTermCfg(
            func=geom_friction, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "ranges": (0.5, 1.5), "operation": "abs", "distribution": "uniform",
                "axes": [0],
            },
        )

    events["reset_base"] = EventTermCfg(
        func=reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "asset_cfg": _ALL_JOINT_CFG,
        },
    )

    events["reset_robot_joints"] = EventTermCfg(
        func=_reset_joints_by_scale, mode="reset",
        params={
            "asset_cfg": _LEG_JOINT_CFG,
            "position_scale_range": (0.5, 1.5),
        },
    )

    # CRITICAL: set PD position targets to the default joint pose so the
    # position-servo actuators do not fight to reach 0 rad during the
    # mj_forward() call that follows reset events.
    events["apply_default_joint_pos_target"] = EventTermCfg(
        func=_apply_default_joint_pos_target, mode="reset",
        params={"asset_cfg": _ALL_JOINT_CFG},
    )

    if not play:
        events["base_external_force_torque"] = EventTermCfg(
            func=apply_external_force_torque, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "force_range": (-10.0, 10.0), "torque_range": (-10.0, 10.0),
            },
        )
        events["push_robot"] = EventTermCfg(
            func=push_by_setting_velocity, mode="interval", interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)}},
        )

    # ---- Rewards ----
    std = math.sqrt(0.25)
    rewards = {
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": std},
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": std},
        ),
        "lin_vel_z_l2": RewardTermCfg(func=lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=ang_vel_xy_l2, weight=-0.05),
        "flat_orientation_l2": RewardTermCfg(func=flat_orientation_l2, weight=-1.0),
        "base_height_l2": RewardTermCfg(func=base_height_l2, weight=-1.0, params={"target_height": 0.363}),
        "joint_torques_l2": RewardTermCfg(func=joint_torques_l2, weight=0.0, params={"asset_cfg": _LEG_ACTUATOR_CFG}),
        "joint_vel_l2": RewardTermCfg(func=joint_vel_l2, weight=0.0, params={"asset_cfg": _LEG_JOINT_CFG}),
        "joint_vel_wheel_l2": RewardTermCfg(
            func=joint_vel_wheel_l2, weight=-0.01,
            params={"asset_cfg": _WHEEL_JOINT_CFG, "command_name": "base_velocity", "command_threshold": 0.1},
        ),
        "joint_acc_l2": RewardTermCfg(func=joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": _ALL_JOINT_CFG}),
        "joint_pos_limits": RewardTermCfg(func=joint_pos_limits, weight=-10.0, params={"asset_cfg": _LEG_JOINT_CFG}),
        "action_rate_l2": RewardTermCfg(func=action_rate_l2, weight=-0.01),
        "undesired_contacts": RewardTermCfg(
            func=undesired_contacts, weight=-1.0,
            params={"sensor_name": "contact_forces", "body_names": "^(?!.*_foot).*", "threshold": 1.0},
        ),
        "upward": RewardTermCfg(func=upward, weight=0.5),
        "default_joint_l2": RewardTermCfg(
            func=default_joint_l2, weight=-1.0,
            params={"asset_cfg": _LEG_JOINT_CFG, "default_joint_pos_patterns": _TITATIT_DEFAULT_JOINT_POS},
        ),
        "hip_pos": RewardTermCfg(
            func=default_joint_l2, weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
                "default_joint_pos_patterns": _TITATIT_DEFAULT_JOINT_POS,
            },
        ),
        # "foot_mirror": RewardTermCfg(
        #     func=foot_mirror, weight=-0.05,
        #     params={"asset_cfg": _LEG_JOINT_CFG},
        # ),
        # "feet_distance_y_exp": RewardTermCfg(
        #     func=feet_distance_y_exp,
        #     weight=1.0,
        #     params={
        #         "asset_cfg": _FOOT_CFG,
        #         "stance_width": 0.41,
        #         "std": 0.20,
        #     },
        # ),
    }

    # ---- Terminations ----
    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "out_of_terrain_bounds": TerminationTermCfg(
            func=out_of_terrain_bounds, time_out=True,
        ),
        "root_state_nonfinite": TerminationTermCfg(func=root_state_nonfinite),
        "base_contact": TerminationTermCfg(
            func=_base_link_contact,
            params={
                "sensor_name": "contact_forces",
                "threshold": 1.0,
            },
        ),
    }

    # ---- Commands ----
    commands = {
        "base_velocity": UniformVelocityCommandCfg(
            resampling_time_range=(10.0, 10.0),
            entity_name="robot",
            rel_standing_envs=0.02,
            rel_heading_envs=0.5,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi),
            ),
        ),
    }

    # ---- Curriculum ----
    curriculum = {}
    if has_terrain:
        curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_vel, params={"command_name": "base_velocity"},
        )

    curriculum["commands_vel"] = CurriculumTermCfg(
        func=commands_vel_reward_based,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy_exp",
            "reward_scale": 2.0,
            "max_curriculum": 3.0,
            "expansion": 0.5,
            "threshold_ratio": 0.8,
        },
    )

    # ---- Simulation ----
    sim = SimulationCfg(
        # Collection-time friendly solver settings.  The previous 200/100/100
        # settings are very conservative and expensive for large parallel RL.
        # If rough contact becomes unstable, try 120/60/40 before going back to
        # 200/100/100.
        mujoco=MujocoCfg(timestep=0.005, iterations=80, ls_iterations=40, ccd_iterations=20),
        njmax=1500,
        contact_sensor_maxmatch=256,
        # Keep MJLab's dump-heavy NanGuard off during normal training.  The
        # wrapper/normalizer still sanitize non-finite obs/costs.  Re-enable only
        # when actively debugging a NaN reproduction.
        nan_guard=NanGuardCfg(enabled=False, output_dir="/tmp/mjlab/nan_dumps"),
    )

    cfg = ManagerBasedRlEnvCfg(
        scene=scene, observations=observations, actions=actions,
        events=events, rewards=_filter_zero_weight_rewards(rewards), terminations=terminations,
        commands=commands, curriculum=curriculum, sim=sim,
        decimation=4, episode_length_s=20.0,
    )

    if play:
        cfg.scene.num_envs = 50
        cfg.scene.env_spacing = 2.5
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("base_external_force_torque", None)
        cfg.events.pop("push_robot", None)
        cfg.events.pop("randomize_actuator_gains", None)
        cfg.events.pop("add_base_mass", None)
        cfg.events.pop("add_base_com", None)
        cfg.events.pop("robot_geom_friction", None)
        # Track base_link instead of auto-detecting (which can pick a foot/wheel).
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
        cfg.viewer.entity_name = "robot"

    # Attach cost terms so downstream code can read them from the cfg directly.
    cfg.cost_terms = _titatit_flat_cost_terms()

    return cfg


def titatit_flat_play_env_cfg(
    num_envs: int = 50,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Convenience wrapper: TITATIT flat play config (noise/randomization disabled)."""
    return titatit_flat_env_cfg(num_envs=num_envs, play=True, flatten_policy_history=flatten_policy_history)
