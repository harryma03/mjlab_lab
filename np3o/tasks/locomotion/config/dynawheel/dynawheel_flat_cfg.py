# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DYNAWHEEL flat task config.

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

_DYNAWHEEL_XML_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "assets" / "robots" / "dynawheel" / "xml" / "dynawheel.xml"


def _get_dynawheel_spec() -> mujoco.MjSpec:
    """Load the robot spec; mjlab recreates actuators from the config below."""
    spec = mujoco.MjSpec.from_file(str(_DYNAWHEEL_XML_PATH))
    for actuator in tuple(spec.actuators):
        spec.delete(actuator)
    return spec

# Actuator groups — mirrors the IsaacLab ``DDT_DYNAWHEEL_CFG.actuators`` layout.
#
# IsaacLab DCMotorCfg for legs:
#   effort_limit=60.0, saturation_effort=80.0, velocity_limit=20.0,
#   stiffness=60.0, damping=1.5, friction=0.0
# → mjlab DcMotorActuatorCfg(stiffness=60.0, damping=1.5,
#   effort_limit=60.0, saturation_effort=80.0, velocity_limit=20.0)
#
# IsaacLab ImplicitActuatorCfg for wheels:
#   effort_limit_sim=12.0, velocity_limit_sim=30.0, stiffness=0.0,
#   damping=0.5, friction=0.0
# → mjlab IdealPdActuatorCfg(stiffness=0.0, damping=0.5, effort_limit=12.0)

_DYNAWHEEL_LEG_ACTUATOR = DcMotorActuatorCfg(
    target_names_expr=(".*_(HAA|HFE|KFE)",),
    stiffness=103.9,
    damping=7.4,
    effort_limit=200.0,
    saturation_effort=200.0,
    velocity_limit=10.0,
)

_DYNAWHEEL_WHEEL_ACTUATOR = IdealPdActuatorCfg(
    target_names_expr=(".*_WHEEL",),
    stiffness=0.0,
    damping=0.5,
    effort_limit=75.0,
)

_DYNAWHEEL_SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
_DYNAWHEEL_SOFT_VEL_LIMIT_FACTOR = 0.9
_DYNAWHEEL_SOFT_EFFORT_LIMIT_FACTOR = 0.9

# ---- regex patterns (ordered: more specific first) ----
# Joint space
LEG_JOINT_PATTERN = ".*_(HAA|HFE|KFE)"
WHEEL_JOINT_PATTERN = ".*_WHEEL"
ALL_JOINT_PATTERN = ".*"
HIP_JOINT_PATTERN = ".*_HAA"
THIGH_CALF_JOINT_PATTERN = ".*_(HFE|KFE)"

# Actuator space (PD actuator names carry the joint name)
LEG_ACTUATOR_PATTERN = ".*_(HAA|HFE|KFE)"
WHEEL_ACTUATOR_PATTERN = ".*_WHEEL"
ALL_ACTUATOR_PATTERN = ".*"

# ---- cached SceneEntityCfg objects ----
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN, preserve_order=True)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=WHEEL_JOINT_PATTERN, preserve_order=True)
_ALL_JOINT_CFG = SceneEntityCfg("robot", joint_names=ALL_JOINT_PATTERN, preserve_order=True)
_LEG_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=LEG_ACTUATOR_PATTERN, preserve_order=True)
_WHEEL_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=WHEEL_ACTUATOR_PATTERN, preserve_order=True)
_ALL_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=ALL_ACTUATOR_PATTERN, preserve_order=True)

# One source of truth for the standing/default pose.  Reset/init reuses this
# same dict, and cost_default_joint also receives this same dict.
_DYNAWHEEL_DEFAULT_JOINT_POS = {
    ".*_HAA": 0.0,
    ".*_HFE": 0.8,
    ".*_KFE": -1.5,
    ".*_WHEEL": 0.0,
}

# Hard-coded DYNAWHEEL limits matching the patterns above.  These are no longer read
# by the cost functions from MJCF at runtime, which keeps training behaviour
# stable and explicit.
_DYNAWHEEL_JOINT_POS_LIMITS = {
    HIP_JOINT_PATTERN: (-0.50, 0.50),
    ".*_HFE": (-1.57, 4.58),
    ".*_KFE": (-2.74, -0.87),
    WHEEL_JOINT_PATTERN: (-99999.0, 99999.0),
}

_DYNAWHEEL_JOINT_VEL_LIMITS = {
    LEG_JOINT_PATTERN: 10.0,
    WHEEL_JOINT_PATTERN: 32.0,
}

_DYNAWHEEL_JOINT_EFFORT_LIMITS = {
    LEG_JOINT_PATTERN: 200.0,
    WHEEL_JOINT_PATTERN: 75.0,
}

_DYNAWHEEL_ACTUATOR_EFFORT_LIMITS = {
    LEG_ACTUATOR_PATTERN: 200.0,
    WHEEL_ACTUATOR_PATTERN: 75.0,
}

_DYNAWHEEL_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(_DYNAWHEEL_LEG_ACTUATOR, _DYNAWHEEL_WHEEL_ACTUATOR),
    soft_joint_pos_limit_factor=_DYNAWHEEL_SOFT_JOINT_POS_LIMIT_FACTOR,
)

_DYNAWHEEL_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.62),
    joint_pos=_DYNAWHEEL_DEFAULT_JOINT_POS,
    joint_vel={".*": 0.0},
)


def get_dynawheel_cfg() -> EntityCfg:
    """Return a fresh ``EntityCfg`` for the DYNAWHEEL robot."""
    return EntityCfg(
        spec_fn=_get_dynawheel_spec,
        articulation=_DYNAWHEEL_ARTICULATION,
        init_state=_DYNAWHEEL_INIT_STATE,
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


def _dynawheel_flat_cost_terms() -> dict:
    """Return cost terms for DYNAWHEEL NP3O constrained training.

    Five terms matching the IsaacLab ``DYNAWHEELFlatCfg.costs``:

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
                "joint_pos_limit_patterns": _DYNAWHEEL_JOINT_POS_LIMITS,
                "soft_ratio": _DYNAWHEEL_SOFT_JOINT_POS_LIMIT_FACTOR,
            },
        ),
        "torque_limit": CostTermCfg(
            func=cost_torque_limit,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=LEG_ACTUATOR_PATTERN),
                "actuator_effort_limit_patterns": _DYNAWHEEL_ACTUATOR_EFFORT_LIMITS,
                "soft_ratio": _DYNAWHEEL_SOFT_EFFORT_LIMIT_FACTOR,
            },
        ),
        "dof_vel_limits": CostTermCfg(
            func=cost_dof_vel_limits,
            scale=1.0, d_value=0.0, k_value=0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
                "joint_vel_limit_patterns": _DYNAWHEEL_JOINT_VEL_LIMITS,
                "soft_ratio": _DYNAWHEEL_SOFT_VEL_LIMIT_FACTOR,
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
                "default_joint_pos_patterns": _DYNAWHEEL_DEFAULT_JOINT_POS,
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
            horizontal_scale=0.2,
            noise_range=_FLAT_RANDOM_NOISE_RANGE,

        ),
        "waves": HfWaveTerrainCfg(
            proportion=_FLAT_PROP_WAVES,
            horizontal_scale=0.2,
            amplitude_range=_FLAT_WAVE_AMPLITUDE,
            num_waves=_FLAT_WAVE_COUNT,

        ),
        "slopes": HfPyramidSlopedTerrainCfg(
            proportion=_FLAT_PROP_SLOPES,
            horizontal_scale=0.2,
            slope_range=_FLAT_SLOPE_RANGE,
            platform_width=2.0,
        ),
    },
)

# ===========================================================================
# 4. DYNAWHEEL flat environment
# ===========================================================================


def dynawheel_flat_env_cfg(
    num_envs: int = 4096,
    play: bool = False,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Build the DYNAWHEEL velocity-tracking environment configuration.

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
                frame=ObjRef(type="body", name="Base", entity="robot"),
                pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
                ray_alignment="yaw",
                max_distance=5.0,
                exclude_parent_body=True,
            ),
        )

    scene = SceneCfg(
        terrain=terrain_cfg,
        entities={"robot": get_dynawheel_cfg()},
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
                "default_joint_pos_patterns": _DYNAWHEEL_DEFAULT_JOINT_POS,
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
                "default_joint_pos_patterns": _DYNAWHEEL_DEFAULT_JOINT_POS,
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
            func=contact_state, params={"sensor_name": "contact_forces", "body_names": ".*_wheel_motor"},
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

    # ---- Action scales (mirrors DYNAWHEELFlatCfg.control) ----
    _action_scale = 0.25
    _hip_scale_reduction = 0.5

    # ---- Actions ----
    actions = {}
    for leg in ("LF", "LH", "RF", "RH"):
        actions[f"{leg.lower()}_leg_pos"] = JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(f"{leg}_HAA", f"{leg}_HFE", f"{leg}_KFE"),
            scale={
                ".*_HAA": _action_scale * _hip_scale_reduction,
                ".*_HFE": _action_scale,
                ".*_KFE": _action_scale,
            },
            clip={".*": (-100.0, 100.0)},
            preserve_order=True,
        )
        actions[f"{leg.lower()}_foot_vel"] = JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=(f"{leg}_WHEEL",),
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
                "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
                "alpha_range": (-0.05, 0.08), "distribution": "uniform",
            },
        )
        events["add_base_com"] = EventTermCfg(
            func=body_com_offset, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
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
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.2),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
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
                "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
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
        "base_height_l2": RewardTermCfg(func=base_height_l2, weight=-1.0, params={"target_height": 0.62}),
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
            params={"sensor_name": "contact_forces", "body_names": "^(?!.*_wheel_motor).*", "threshold": 1.0},
        ),
        "upward": RewardTermCfg(func=upward, weight=0.5),
        "default_joint_l2": RewardTermCfg(
            func=default_joint_l2, weight=-1.0,
            params={"asset_cfg": _LEG_JOINT_CFG, "default_joint_pos_patterns": _DYNAWHEEL_DEFAULT_JOINT_POS},
        ),
        "hip_pos": RewardTermCfg(
            func=default_joint_l2, weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_HAA"),
                "default_joint_pos_patterns": _DYNAWHEEL_DEFAULT_JOINT_POS,
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
            "max_curriculum": 1.5,
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
        # Track Base instead of auto-detecting (which can pick a foot/wheel).
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
        cfg.viewer.entity_name = "robot"

    # Attach cost terms so downstream code can read them from the cfg directly.
    cfg.cost_terms = _dynawheel_flat_cost_terms()

    return cfg


def dynawheel_flat_play_env_cfg(
    num_envs: int = 50,
    flatten_policy_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Convenience wrapper: DYNAWHEEL flat play config (noise/randomization disabled)."""
    return dynawheel_flat_env_cfg(num_envs=num_envs, play=True, flatten_policy_history=flatten_policy_history)
