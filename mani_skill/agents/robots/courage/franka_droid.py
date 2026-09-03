import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_mjcf_agent import BaseMjcfAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class FrankaDroid(BaseMjcfAgent):
    uid = "franka-droid"
    mjcf_path = f"{PACKAGE_ASSET_DIR}/robots/franka_droid/model.xml"

    @property
    def _sensor_configs(self):
        # # Add custom cameras mounted to a link on the robot, or remove this if there aren't any you wish to simulate
        # # Specify the position of camera with sapien.Pose or the ManiSkill Pose class.
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
        return []
