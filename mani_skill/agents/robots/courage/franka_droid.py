from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_mjcf_agent import BaseMjcfAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.pd_joint_pos import MimicConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class FrankaDroid(BaseMjcfAgent):
    uid = "franka-droid"
    mjcf_path = f"{PACKAGE_ASSET_DIR}/robots/franka_droid/model.xml"
    disable_self_collisions = True

    @property
    def _controller_configs(self):
        passive_finger_joint_names = [
            "gripper/left_spring_link_joint",
            "gripper/left_follower",
            "gripper/right_spring_link_joint",
            "gripper/right_follower_joint",
        ]

        passive_finger_joints_controllers = PassiveControllerConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=0,
        )

        driver_joint_controller = PDJointPosMimicControllerConfig(
            joint_names=["gripper/left_driver_joint", "gripper/right_driver_joint"],
            lower=None,
            upper=None,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            friction=0.05,
            normalize_action=False,
            mimic={
                "gripper/right_driver_joint": MimicConfig(
                    joint="gripper/left_driver_joint",
                    multiplier=1.0,
                    offset=0.0,
                ),
                # "gripper/left_spring_link_joint": MimicConfig(
                #     joint="gripper/left_driver_joint",
                #     multiplier=1.0,
                #     offset=0.0,
                # ),
                # "gripper/left_follower": MimicConfig(
                #     joint="gripper/left_driver_joint",
                #     multiplier=1.0,
                #     offset=0.0,
                # ),
                # "gripper/right_spring_link_joint": MimicConfig(
                #     joint="gripper/left_driver_joint",
                #     multiplier=1.0,
                #     offset=0.0,
                # ),
                # "gripper/right_follower_joint": MimicConfig(
                #     joint="gripper/left_driver_joint",
                #     multiplier=1.0,
                #     offset=0.0,
                # ),
            },
        )

        self.mjcf_controllers.setdefault("pd_joint_pos", {}).update(
            {
                "passive_finger_joints": passive_finger_joints_controllers,
                "driver_joint_controller": driver_joint_controller,
            }
        )

        return self.mjcf_controllers

    @property
    def _sensor_configs(self):
        cameras: list[CameraConfig] = []

        # return [
        #     CameraConfig(
        #         uid="your_custom_camera_on_this_robot",
        #         pose=sapien.Pose(
        #             p=[0.0464982, -0.0200011, 0.0360011],
        #             q=[0, 0.70710678, 0, 0.70710678],
        #         ),
        #         width=128,
        #         height=128,
        #         fov=1.57,
        #         near=0.01,
        #         far=100,
        #         entity_uid="your_mounted_camera",
        #     )
        # ]

        return cameras
