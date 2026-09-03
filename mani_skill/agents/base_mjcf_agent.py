from pathlib import Path

import mujoco as mj
import sapien

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.loaders.mjcf.articulation_loader import MjcfAssetArticulationLoader
from mani_skill.loaders.mjcf.common import parse_xml
from mani_skill.utils.structs import Pose


def is_position_actuator(act: mj.MjsActuator) -> bool:
    return (act.dyntype, act.gaintype, act.biastype) == (
        mj.mjtDyn.mjDYN_NONE,
        mj.mjtGain.mjGAIN_FIXED,
        mj.mjtBias.mjBIAS_AFFINE,
    )


class BaseMjcfAgent(BaseAgent):
    def __init__(
        self,
        scene: ManiSkillScene,
        control_freq: int,
        control_mode: str | None = None,
        agent_idx: int | None = None,
        initial_pose: sapien.Pose | Pose | None = None,
        build_separate: bool = False,
    ) -> None:
        self._mjcf_articulation_loader = MjcfAssetArticulationLoader()

        self._mjcf_controllers = {}

        super().__init__(
            scene, control_freq, control_mode, agent_idx, initial_pose, build_separate
        )

    def _load_articulation(
        self, initial_pose: sapien.Pose | Pose | None = None
    ) -> None:
        self._mjcf_articulation_loader.set_scene(self.scene)

        assert (
            self.mjcf_path is not None
        ), "Must provide a mjcf model for this type of agent"

        mjcf_filepath = Path(self.mjcf_path)
        assert (
            mjcf_filepath.is_file()
        ), f"Given mjcf file @ {mjcf_filepath} doesn't exist"

        spec, folders = parse_xml(mjcf_filepath)

        builder = self._mjcf_articulation_loader.load_from_spec(
            spec,
            mjcf_filepath.parent,
            floating_base=not self.fix_root_link,
            folders=folders,
        )
        builder.initial_pose = initial_pose
        builder.set_name(self.uid)
        if self._agent_idx is not None:
            builder.set_name(f"{self.uid}-agent-{self._agent_idx}")

        self.robot = builder.build()
        self.robot_link_names = [link.name for link in self.robot.get_links()]

        for actuator_spec in spec.actuators:
            assert isinstance(actuator_spec, mj.MjsActuator)
            actuator_name = (
                actuator_spec.name if actuator_spec.name != "" else actuator_spec.target
            )
            if is_position_actuator(actuator_spec):
                kp = actuator_spec.gainprm[0].item()
                kv = -actuator_spec.biasprm[2].item()
                pd_controller = PDJointPosControllerConfig(
                    [actuator_spec.target],
                    lower=None,
                    upper=None,
                    stiffness=kp,
                    damping=kv,
                    normalize_action=False,
                )
                if "pd_joint_pos" not in self._mjcf_controllers:
                    self._mjcf_controllers["pd_joint_pos"] = {}
                self._mjcf_controllers["pd_joint_pos"][actuator_name] = pd_controller

    @property
    def _controller_configs(self):
        return self._mjcf_controllers
