"""ZG-M1 robot contract shared by the flat and rough velocity tasks."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.actuator.dc_actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ObjRef

from ...cmdp.costs import cost_default_joint, cost_dof_vel_limits, cost_hip_pos, cost_pos_limit, cost_torque_limit


_ZG_M1_XML_PATH = Path(__file__).parents[5] / "assets" / "robots" / "zg_m1" / "mujoco" / "zg_m1.xml"

LEG_JOINT_PATTERN = ".*_(ABAD|HIP|KNEE)_JOINT"
WHEEL_JOINT_PATTERN = ".*_FOOT_JOINT"
ABAD_JOINT_PATTERN = ".*_ABAD_JOINT"
HIP_KNEE_JOINT_PATTERN = ".*_(HIP|KNEE)_JOINT"

_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN, preserve_order=True)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=WHEEL_JOINT_PATTERN, preserve_order=True)
_ALL_JOINT_CFG = SceneEntityCfg("robot", joint_names=".*", preserve_order=True)
_LEG_ACTUATOR_CFG = SceneEntityCfg("robot", actuator_names=LEG_JOINT_PATTERN, preserve_order=True)

_DEFAULT_JOINT_POS = {
    "FAR_ABAD_JOINT": 0.0, "FAR_HIP_JOINT": 0.8, "FAR_KNEE_JOINT": -1.5, "FAR_FOOT_JOINT": 0.0,
    "FBL_ABAD_JOINT": 0.0, "FBL_HIP_JOINT": 0.8, "FBL_KNEE_JOINT": -1.5, "FBL_FOOT_JOINT": 0.0,
    "RAR_ABAD_JOINT": 0.0, "RAR_HIP_JOINT": -0.8, "RAR_KNEE_JOINT": 1.5, "RAR_FOOT_JOINT": 0.0,
    "RBL_ABAD_JOINT": 0.0, "RBL_HIP_JOINT": -0.8, "RBL_KNEE_JOINT": 1.5, "RBL_FOOT_JOINT": 0.0,
}

_JOINT_POS_LIMITS = {
    "FAR_ABAD_JOINT": (-0.697, 0.523), "RAR_ABAD_JOINT": (-0.697, 0.523),
    "FBL_ABAD_JOINT": (-0.523, 0.697), "RBL_ABAD_JOINT": (-0.523, 0.697),
    ".*_HIP_JOINT": (-2.443, 2.443), ".*_KNEE_JOINT": (-2.801, 2.801),
    WHEEL_JOINT_PATTERN: (-1.0e6, 1.0e6),
}
_JOINT_VEL_LIMITS = {LEG_JOINT_PATTERN: 16.75, WHEEL_JOINT_PATTERN: 110.0}
_ACTUATOR_EFFORT_LIMITS = {LEG_JOINT_PATTERN: 180.0, WHEEL_JOINT_PATTERN: 28.68}

_ABAD_ACTUATOR = DcMotorActuatorCfg(
    target_names_expr=(ABAD_JOINT_PATTERN,), stiffness=60.0, damping=1.0,
    effort_limit=180.0, saturation_effort=180.0, velocity_limit=16.75,
)
_HIP_KNEE_ACTUATOR = DcMotorActuatorCfg(
    target_names_expr=(HIP_KNEE_JOINT_PATTERN,), stiffness=120.0, damping=1.0,
    effort_limit=180.0, saturation_effort=180.0, velocity_limit=16.75,
)
_WHEEL_ACTUATOR = IdealPdActuatorCfg(
    target_names_expr=(WHEEL_JOINT_PATTERN,), stiffness=0.0, damping=0.5,
    effort_limit=28.68,
)


def _get_zg_m1_spec() -> mujoco.MjSpec:
    """Load the generated floating-base MJCF model."""
    return mujoco.MjSpec.from_file(str(_ZG_M1_XML_PATH))


def get_zg_m1_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_zg_m1_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(_ABAD_ACTUATOR, _HIP_KNEE_ACTUATOR, _WHEEL_ACTUATOR),
            soft_joint_pos_limit_factor=0.9,
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.55), joint_pos=_DEFAULT_JOINT_POS, joint_vel={".*": 0.0},
        ),
        sort_actuators=True,
    )


def _cost_terms() -> dict:
    from np3o.algorithms.np3o.cost_manager import CostTermCfg

    return {
        "pos_limit": CostTermCfg(func=cost_pos_limit, scale=1.0, d_value=0.0, k_value=0.01, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
            "joint_pos_limit_patterns": _JOINT_POS_LIMITS, "soft_ratio": 0.9,
        }),
        "torque_limit": CostTermCfg(func=cost_torque_limit, scale=1.0, d_value=0.0, k_value=0.01, params={
            "asset_cfg": SceneEntityCfg("robot", actuator_names=LEG_JOINT_PATTERN),
            "actuator_effort_limit_patterns": _ACTUATOR_EFFORT_LIMITS, "soft_ratio": 0.9,
        }),
        "dof_vel_limits": CostTermCfg(func=cost_dof_vel_limits, scale=1.0, d_value=0.0, k_value=0.01, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_PATTERN),
            "joint_vel_limit_patterns": _JOINT_VEL_LIMITS, "soft_ratio": 0.9,
        }),
        "hip_pos": CostTermCfg(func=cost_hip_pos, scale=2.0, d_value=0.0, k_value=0.01, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ABAD_JOINT_PATTERN),
        }),
        "default_joint": CostTermCfg(func=cost_default_joint, scale=0.2, d_value=0.0, k_value=0.01, params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HIP_KNEE_JOINT_PATTERN),
            "default_joint_pos_patterns": _DEFAULT_JOINT_POS,
        }),
    }


def _apply_default_joint_pos_target(env, env_ids, asset_cfg: SceneEntityCfg = _ALL_JOINT_CFG) -> None:
    asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    asset.set_joint_position_target(asset.data.joint_pos[:, asset_cfg.joint_ids], joint_ids=asset_cfg.joint_ids)


def configure_zg_m1(base_cfg):
    """Replace D1's robot-specific contract while retaining its task settings."""
    base_cfg.scene.entities["robot"] = get_zg_m1_cfg()
    for sensor in base_cfg.scene.sensors:
        if sensor.name == "height_scanner":
            sensor.frame = ObjRef(type="body", name="BASE_LINK", entity="robot")

    base_cfg.actions = {}
    for leg in ("FAR", "FBL", "RAR", "RBL"):
        base_cfg.actions[f"{leg.lower()}_leg_pos"] = JointPositionActionCfg(
            entity_name="robot", actuator_names=(f"{leg}_ABAD_JOINT", f"{leg}_HIP_JOINT", f"{leg}_KNEE_JOINT"),
            scale=0.25, clip={".*": (-100.0, 100.0)}, preserve_order=True,
        )
        base_cfg.actions[f"{leg.lower()}_foot_vel"] = JointVelocityActionCfg(
            entity_name="robot", actuator_names=(f"{leg}_FOOT_JOINT",),
            scale=5.0, clip={".*": (-100.0, 100.0)}, preserve_order=True,
        )

    for group in ("policy", "critic"):
        base_cfg.observations[group].terms["joint_pos"].params.update({
            "asset_cfg": _ALL_JOINT_CFG, "wheel_asset_cfg": _WHEEL_JOINT_CFG,
            "default_joint_pos_patterns": _DEFAULT_JOINT_POS,
        })
        base_cfg.observations[group].terms["joint_vel"].params["asset_cfg"] = _ALL_JOINT_CFG
    base_cfg.observations["priv"].terms["contact_state"].params["body_names"] = ".*_FOOT_LINK"
    for name in ("joint_kp_factor", "joint_kd_factor"):
        base_cfg.observations["priv"].terms[name].params["asset_cfg"] = _ALL_JOINT_CFG

    base_cfg.events["reset_base"].params["asset_cfg"] = _ALL_JOINT_CFG
    base_cfg.events["reset_robot_joints"].params["asset_cfg"] = _LEG_JOINT_CFG
    base_cfg.events["apply_default_joint_pos_target"].func = _apply_default_joint_pos_target
    base_cfg.events["apply_default_joint_pos_target"].params["asset_cfg"] = _ALL_JOINT_CFG
    for name in ("add_base_mass", "add_base_com", "base_external_force_torque"):
        if name in base_cfg.events:
            base_cfg.events[name].params["asset_cfg"] = SceneEntityCfg("robot", body_names="BASE_LINK")

    rewards = base_cfg.rewards
    if "base_height_l2" in rewards:
        rewards["base_height_l2"].params["target_height"] = 0.50
    if "joint_torques_l2" in rewards:
        rewards["joint_torques_l2"].params["asset_cfg"] = _LEG_ACTUATOR_CFG
    for name in ("joint_vel_l2", "joint_pos_limits", "default_joint_l2"):
        if name in rewards:
            rewards[name].params["asset_cfg"] = _LEG_JOINT_CFG
    if "joint_vel_wheel_l2" in rewards:
        rewards["joint_vel_wheel_l2"].params["asset_cfg"] = _WHEEL_JOINT_CFG
    if "joint_acc_l2" in rewards:
        rewards["joint_acc_l2"].params["asset_cfg"] = _ALL_JOINT_CFG
    if "undesired_contacts" in rewards:
        rewards["undesired_contacts"].params["body_names"] = "^(?!.*_FOOT_LINK).*"
    if "default_joint_l2" in rewards:
        rewards["default_joint_l2"].params["default_joint_pos_patterns"] = _DEFAULT_JOINT_POS
    if "hip_pos" in rewards:
        rewards["hip_pos"].params.update({
            "asset_cfg": SceneEntityCfg("robot", joint_names=ABAD_JOINT_PATTERN),
            "default_joint_pos_patterns": _DEFAULT_JOINT_POS,
        })
    base_cfg.cost_terms = _cost_terms()
    return base_cfg
