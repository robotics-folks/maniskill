from pathlib import Path

import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.loaders.mjcf.actor_loader import MjcfAssetActorLoader
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env


@register_env("EmptyMjcf-v1", max_episode_steps=20000)
class EmptyMjcfEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ("none",)

    def __init__(self, *args, model_path: Path | None = None, **kwargs) -> None:
        self._mjcf_actor_loader = MjcfAssetActorLoader()
        self._model_path = model_path

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
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)

        self._mjcf_actor_loader.set_scene(self.scene)

        if self._model_path and self._model_path.is_file():
            builder = self._mjcf_actor_loader.load_from_xml(self._model_path)
            builder.build(self._model_path.stem)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return {}
