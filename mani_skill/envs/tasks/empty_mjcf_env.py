# pyright: reportUnusedImport=false

from pathlib import Path
from pprint import pprint

import numpy as np
import torch

from mani_skill.agents.robots.courage import FrankaDroid  # noqa: F401
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.loaders.mjcf.actor_loader import MjcfAssetActorLoader
from mani_skill.loaders.mjcf.articulation_loader import MjcfAssetArticulationLoader
from mani_skill.loaders.mjcf.common import is_asset_articulated
from mani_skill.loaders.mjcf.scene_loader import MjcfSceneLoader
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env

ORIGINAL_TO_MUJOCO_SHADERS: dict[str, str] = {
    "default": "default-mj",
    "rt-fast": "rt-fast-mj",
}


@register_env("EmptyMjcf-v1", max_episode_steps=20000)
class EmptyMjcfEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ("none",)

    def __init__(
        self,
        *args,
        model_path: Path | None = None,
        scene_path: Path | None = None,
        **kwargs,
    ) -> None:
        self._mjcf_actor_loader = MjcfAssetActorLoader()
        self._mjcf_articulation_loader = MjcfAssetArticulationLoader()
        self._mjcf_scene_loader = MjcfSceneLoader()

        if (sensor_cfgs := kwargs.get("sensor_configs")) and (
            shader_pack := sensor_cfgs.get("shader_pack")
        ) in ORIGINAL_TO_MUJOCO_SHADERS:
            sensor_cfgs["shader_pack"] = ORIGINAL_TO_MUJOCO_SHADERS[shader_pack]

        if (viewer_cam_cfgs := kwargs.get("viewer_camera_configs")) and (
            shader_pack := viewer_cam_cfgs.get("shader_pack")
        ) in ORIGINAL_TO_MUJOCO_SHADERS:
            viewer_cam_cfgs["shader_pack"] = ORIGINAL_TO_MUJOCO_SHADERS[shader_pack]

        self._model_path = model_path
        self._scene_path = scene_path

        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self._mjcf_actor_loader.set_scene(self.scene)
        self._mjcf_articulation_loader.set_scene(self.scene)
        self._mjcf_scene_loader.set_scene(self.scene)

        if self._model_path and self._model_path.is_file():
            if is_asset_articulated(self._model_path):
                builder = self._mjcf_articulation_loader.load_from_xml(self._model_path)
            else:
                builder = self._mjcf_actor_loader.load_from_xml(self._model_path)
            builder.build(self._model_path.stem)

        make_ground = True
        if self._scene_path and self._scene_path.is_file():
            actors, articulations = self._mjcf_scene_loader.load(
                self._scene_path, add_ground=True, add_lights=False
            )
            make_ground = False

            print("Actors:")
            pprint(list(actors.keys()))
            print("-" * 20)
            print("Articulation:")
            pprint(list(articulations.keys()))

        if make_ground:
            self.ground = build_ground(self.scene)
            self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return {}
