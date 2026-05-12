# pyright: reportCallIssue=false
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr.ik.robot import BaseRobot, Cost


# Actuated joint name groups
_ARM_L_JOINTS = [f"arm_l_joint{i}" for i in range(1, 8)]
_ARM_R_JOINTS = [f"arm_r_joint{i}" for i in range(1, 8)]
_LIFT_JOINT = "lift_joint"
_SWERVE_JOINTS = [
    "front_left_steer_joint",
    "rear_left_steer_joint",
    "rear_right_steer_joint",
    "front_right_steer_joint",
    "front_left_drive_joint",
    "rear_left_drive_joint",
    "rear_right_drive_joint",
    "front_right_drive_joint",
]

# DEUX description package root (relative to this file's repo location)
_DEUX_DESCRIPTION_ROOT = (
    Path(__file__).resolve().parents[5] / "deux_description"
)
_DEUX_URDF_PATH = _DEUX_DESCRIPTION_ROOT / "urdf" / "deux" / "deux.urdf"


class DEUX(BaseRobot):
    """
    DEUX bimanual mobile robot IK implementation.

    Structure:
      - base_link -> lift_joint (prismatic) -> upper_body
      - upper_body -> arm_l_joint1..7 -> left_arm_hand  (L ee)
      - upper_body -> arm_r_joint1..7 -> right_arm_hand (R ee)
      - base_link  -> swerve drive joints (locked at 0 during IK)
    """

    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        super().__init__()
        urdf = self._load_urdf(urdf_string)
        self.robot: pk.Robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        self.L_ee = "left_arm_hand"
        self.R_ee = "right_arm_hand"
        self.height_link = "upper_body"

        names = self.robot.links.names

        if self.L_ee not in names:
            raise ValueError(f"Link {self.L_ee} not found in URDF")
        if self.R_ee not in names:
            raise ValueError(f"Link {self.R_ee} not found in URDF")
        if self.height_link not in names:
            raise ValueError(f"Link {self.height_link} not found in URDF")

        self.L_ee_link_idx: int = names.index(self.L_ee)
        self.R_ee_link_idx: int = names.index(self.R_ee)
        self.height_link_idx: int = names.index(self.height_link)

        joint_names = list(self.robot.joints.actuated_names)
        self._swerve_indices = [
            i for i, n in enumerate(joint_names) if n in _SWERVE_JOINTS
        ]
        self._arm_l_indices = [
            i for i, n in enumerate(joint_names) if n in _ARM_L_JOINTS
        ]
        self._arm_r_indices = [
            i for i, n in enumerate(joint_names) if n in _ARM_R_JOINTS
        ]

    @override
    def _load_default_urdf(self) -> yourdfpy.URDF:
        if not _DEUX_URDF_PATH.exists():
            raise FileNotFoundError(
                f"DEUX URDF not found at {_DEUX_URDF_PATH}. "
                "Ensure deux_description is cloned next to teleop_xr."
            )
        self.urdf_path = str(_DEUX_URDF_PATH)
        self.mesh_path = str(_DEUX_DESCRIPTION_ROOT / "meshes")
        return yourdfpy.URDF.load(
            self.urdf_path,
            load_meshes=False,
            load_collision_meshes=False,
        )

    @property
    @override
    def supported_frames(self) -> set[str]:
        return {"left", "right", "head"}

    @property
    @override
    def joint_var_cls(self) -> Any:
        return self.robot.joint_var_cls

    @property
    @override
    def actuated_joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    @override
    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(config)
        return {
            "left": jaxlie.SE3(fk[self.L_ee_link_idx]),
            "right": jaxlie.SE3(fk[self.R_ee_link_idx]),
            "head": jaxlie.SE3(fk[self.height_link_idx]),
        }

    @override
    def get_default_config(self) -> jax.Array:
        defaults = {
            "arm_l_joint1": 0.0,
            "arm_l_joint2": -0.8,
            "arm_l_joint3": 0.0,
            "arm_l_joint4": 1.2,
            "arm_l_joint5": 0.0,
            "arm_l_joint6": 0.0,
            "arm_l_joint7": 0.0,
            "arm_r_joint1": 0.0,
            "arm_r_joint2": 0.8,
            "arm_r_joint3": 0.0,
            "arm_r_joint4": 1.2,
            "arm_r_joint5": 0.0,
            "arm_r_joint6": 0.0,
            "arm_r_joint7": 0.0,
            _LIFT_JOINT: 0.0,
        }
        return jnp.array(
            [defaults.get(n, 0.0) for n in self.actuated_joint_names]
        )

    @override
    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        costs = []
        JointVar = self.robot.joint_var_cls
        n_joints = len(self.actuated_joint_names)

        # Rest cost: stay near current config for arm joints
        if q_current is not None:
            rest_weights = jnp.zeros(n_joints)
            for i in self._arm_l_indices + self._arm_r_indices:
                rest_weights = rest_weights.at[i].set(5.0)
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=rest_weights,
                )
            )

        # Lock swerve drive joints at 0
        if self._swerve_indices:
            swerve_weights = jnp.zeros(n_joints)
            for i in self._swerve_indices:
                swerve_weights = swerve_weights.at[i].set(100.0)
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=jnp.zeros(n_joints),
                    weight=swerve_weights,
                )
            )

        # Manipulability cost for arms
        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32),
                weight=0.005,
            )
        )

        # Left EE pose cost
        if target_L is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_L,
                    jnp.array(self.L_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        # Right EE pose cost
        if target_R is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_R,
                    jnp.array(self.R_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        # Head/lift height cost
        if target_Head is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_Head,
                    jnp.array(self.height_link_idx, dtype=jnp.int32),
                    pos_weight=20.0,
                    ori_weight=0.0,
                )
            )

        # Joint limit cost
        costs.append(pk.costs.limit_cost(self.robot, JointVar(0), weight=50.0))

        # Self-collision cost
        costs.append(
            pk.costs.self_collision_cost(
                self.robot,
                self.robot_coll,
                JointVar(0),
                margin=0.05,
                weight=10.0,
            )
        )

        return costs
