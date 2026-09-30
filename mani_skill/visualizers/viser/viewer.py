import numpy as np
import viser

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation


class ViserViewer:
    def __init__(self, scene: ManiSkillScene) -> None:
        self._scene = scene

    def _build_viser_scene(self) -> None:
        pass

    def _build_from_actor(self, actor: Actor) -> None:
        pass

    @property
    def scene(self) -> ManiSkillScene:
        return self._scene
