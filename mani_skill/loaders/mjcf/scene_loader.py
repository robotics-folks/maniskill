from __future__ import annotations

from pathlib import Path

import mujoco as mj
from sapien import Pose

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs import Actor, Articulation

from .actor_loader import MjcfAssetActorLoader
from .articulation_loader import MjcfAssetArticulationLoader
from .common import get_orientation, has_any_non_free_joint


class MjcfSceneLoader:
    def __init__(self, scene: ManiSkillScene | None = None) -> None:
        self._scene: ManiSkillScene | None = scene
        self._spec: mj.MjSpec | None = None
        self._num_actors: int = 0
        self._num_articulations: int = 0

    def set_scene(self, scene: ManiSkillScene) -> MjcfSceneLoader:
        self._scene = scene
        return self

    @property
    def mjspec(self) -> mj.MjSpec | None:
        return self._spec

    def load(
        self,
        scene_path: Path,
        add_ground: bool = True,
        add_lights: bool = True,
    ) -> tuple[dict[str, Actor], dict[str, Articulation]]:
        assert (
            self._scene is not None
        ), "Must assign a valid Sapien scene before loading"

        actors: dict[str, Actor] = {}
        articulations: dict[str, Articulation] = {}

        articulation_loader = MjcfAssetArticulationLoader(self._scene)
        actor_loader = MjcfAssetActorLoader(self._scene)

        if add_ground:
            for subscene in self._scene.sub_scenes:
                subscene.add_ground(altitude=0, render=False)

        spec = mj.MjSpec.from_file(scene_path.as_posix())
        for root_body in spec.worldbody.bodies:
            assert isinstance(root_body, mj.MjsBody)

            world_pose = Pose(
                p=tuple(root_body.pos), q=tuple(get_orientation(root_body))
            )
            if has_any_non_free_joint(root_body):
                name = (
                    root_body.name
                    if root_body.name != ""
                    else f"articulation_{self._num_articulations}"
                )
                builder = articulation_loader.load_from_spec(
                    scene_spec=spec,
                    model_dir=scene_path.parent,
                    root_body_name=root_body.name,
                    is_part_of_scene=True,
                )
                builder.set_initial_pose(world_pose)
                articulations[name] = builder.build(name=name)
                self._num_articulations += 1
            else:
                if root_body.name == "floor":
                    continue
                name = (
                    root_body.name
                    if root_body.name != ""
                    else f"actor_{self._num_actors}"
                )
                builder = actor_loader.load_from_spec(
                    scene_spec=spec,
                    model_dir=scene_path.parent,
                    root_body_name=root_body.name,
                )
                builder.set_name(name)
                builder.set_initial_pose(world_pose)
                actors[name] = builder.build(name)
                self._num_actors += 1

        if add_lights:
            for light in spec.worldbody.lights:
                assert isinstance(light, mj.MjsLight)
                match light.type:
                    case mj.mjtLightType.mjLIGHT_DIRECTIONAL:
                        self._scene.add_directional_light(
                            direction=light.dir,
                            color=light.diffuse,
                            shadow=bool(light.castshadow),
                        )
                    case mj.mjtLightType.mjLIGHT_POINT:
                        pass
                    case _:
                        pass

        return actors, articulations
