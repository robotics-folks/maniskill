import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np
import sapien
from sapien.render import RenderMaterial, RenderTexture2D
from scipy.spatial.transform import Rotation as R

VISUAL_CLASSES = {"__VISUAL_MJT__", "visual"}
VISUAL_THRESHOLD_DENSITY = 1e-5
VISUAL_THRESHOLD_MASS = 1e-6

CAPSULE_FIX_POSE = sapien.Pose(
    q=R.from_euler("xyz", [0, np.pi / 2, 0]).as_quat(scalar_first=True)
)
CYLINDER_FIX_POSE = sapien.Pose(
    q=R.from_euler("xyz", [0, np.pi / 2, 0]).as_quat(scalar_first=True)
)

X_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

WORLD_UP = Z_AXIS.copy()

THOR_COLLISION_GROUPS: dict[str, list[int]] = {
    "__STRUCTURAL_MJT__": [0b1000, 0b1111, 1, 0],
    "__STRUCTURAL_WALL_MJT__": [0b1000, 0b1111, 1, 0],
    # "__DYNAMIC_MJT__": [0b0001, 0b1111, 1, 0],
    "__ARTICULABLE_DYNAMIC_MJT__": [0b0000, 0b0111, 1, 0],
}


@dataclass
class MjcfAssetsFolders:
    textures: dict[str, Path] = field(default_factory=dict)
    meshes: dict[str, Path] = field(default_factory=dict)

    def get_best_texture_match(self, name: str) -> Path | None:
        for tex_name, tex_path in self.textures.items():
            if tex_name in name:
                return tex_path

    def get_best_mesh_match(self, name: str) -> Path | None:
        for mesh_name, mesh_path in self.meshes.items():
            if mesh_name in name:
                return mesh_path


@dataclass
class MjcfTextureInfo:
    name: str
    type: mj.mjtTexture
    rgb1: list
    rgb2: list
    file: Path


@dataclass
class MjcfJointInfo:
    name: str
    type: Literal["free", "fixed", "hinge", "slide", "ball"]
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=X_AXIS.copy)
    limited: bool = False
    limits: np.ndarray = field(default_factory=lambda: np.array([-np.inf, np.inf]))
    frictionloss: float = 0.0
    damping: float = 0.0


def mjc_joint_type_to_str(
    jnt_type: mj.mjtJoint,
) -> Literal["free", "fixed", "hinge", "slide", "ball"]:
    match jnt_type:
        case mj.mjtJoint.mjJNT_FREE:
            return "free"
        case mj.mjtJoint.mjJNT_HINGE:
            return "hinge"
        case mj.mjtJoint.mjJNT_SLIDE:
            return "slide"
        case mj.mjtJoint.mjJNT_BALL:
            return "ball"
        case _:
            raise RuntimeError(f"Joint of type '{jnt_type}' is not valid")


QUAT_TOLERANCE = 1e-10
AXIS_NORM_TOLERANCE = 1e-3


def vec_to_quat(vec: np.ndarray) -> np.ndarray:
    vec /= np.linalg.norm(vec)

    cross = np.cross(Z_AXIS, vec)
    s = np.linalg.norm(cross)

    if s < QUAT_TOLERANCE:
        return np.array([0.0, 1.0, 0.0, 0.0])
    else:
        cross /= np.linalg.norm(cross)
        ang = np.arctan2(s, vec[2]).item()
        quat = np.array(
            [
                np.cos(ang / 2.0),
                cross[0] * np.sin(ang / 2.0),
                cross[1] * np.sin(ang / 2.0),
                cross[2] * np.sin(ang / 2.0),
            ]
        )
        quat /= np.linalg.norm(quat)
        return quat


def get_frame_axes(axis_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(np.dot(axis_x, [1.0, 0.0, 0.0])) > 0.9:
        axis_y = np.cross(axis_x, WORLD_UP)
        axis_y = axis_y / np.linalg.norm(axis_y)
    else:
        axis_y = np.cross(axis_x, X_AXIS)
        axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    axis_z = axis_z / np.linalg.norm(axis_z)

    return axis_x, axis_y, axis_z


def is_visual(geom: mj.MjsGeom) -> bool:
    return (geom.classname.name in VISUAL_CLASSES) or (
        geom.contype == 0 and geom.conaffinity == 0
    )


def get_visual_specs_from_body(mjs_body: mj.MjsBody) -> list[mj.MjsGeom]:
    return [geom for geom in mjs_body.geoms if is_visual(geom)]


def get_collider_specs_from_body(mjs_body: mj.MjsBody) -> list[mj.MjsGeom]:
    return [geom for geom in mjs_body.geoms if not is_visual(geom)]


def get_orientation(obj_spec: mj.MjsBody | mj.MjsGeom | mj.MjsFrame) -> np.ndarray:
    match obj_spec.alt.type:
        case mj.mjtOrientation.mjORIENTATION_QUAT:
            return obj_spec.quat.copy()
        case mj.mjtOrientation.mjORIENTATION_AXISANGLE:
            axisangle = obj_spec.alt.axisangle
            return R.from_rotvec(axisangle[-1] * axisangle[:-1]).as_quat(
                scalar_first=True
            )
        case mj.mjtOrientation.mjORIENTATION_XYAXES:
            raise NotImplementedError(
                "Support for xyaxes in orientation is not supported yet"
            )
        case mj.mjtOrientation.mjORIENTATION_ZAXIS:
            raise NotImplementedError(
                "Support for zaxis in orientation is not supported yet"
            )
        case mj.mjtOrientation.mjORIENTATION_EULER:
            euler = obj_spec.alt.euler
            return R.from_euler("xyz", euler, degrees=False).as_quat(scalar_first=True)
        case _:
            raise ValueError(
                f"Orientation type {obj_spec.alt.type} is not valid, for obj: {obj_spec.name}"
            )


def get_rgba_from_geom(mj_spec: mj.MjSpec, mj_geom: mj.MjsGeom) -> np.ndarray:
    rgba = mj_geom.rgba.copy()
    if mj_geom.material:
        mj_mat = mj_spec.material(mj_geom.material)
        if mj_mat is not None:
            rgba = mj_mat.rgba.copy()
    return rgba


def parse_textures(
    spec: mj.MjSpec, folders: MjcfAssetsFolders | None = None
) -> dict[str, MjcfTextureInfo]:
    textures_info: dict[str, MjcfTextureInfo] = {}
    for tex_spec in spec.textures:
        assert isinstance(tex_spec, mj.MjsTexture)
        if tex_spec.name in textures_info:
            print(f"[WARN]: texture with name {tex_spec.name} already parsed")
            continue
        path_str = os.path.join(spec.modelfiledir, spec.texturedir, tex_spec.file)
        tex_path = Path(os.path.relpath(path_str))
        if folders:
            tex_name = tex_spec.name if tex_spec.name != "" else tex_path.stem
            if ovr_tex_path := folders.get_best_texture_match(tex_name):
                tex_path = ovr_tex_path
        textures_info[tex_spec.name] = MjcfTextureInfo(
            name=tex_spec.name,
            type=tex_spec.type,
            rgb1=tex_spec.rgb1.tolist(),
            rgb2=tex_spec.rgb2.tolist(),
            file=tex_path,
        )

    return textures_info


def parse_materials(
    spec: mj.MjSpec, folders: MjcfAssetsFolders | None = None
) -> dict[str, RenderMaterial]:
    textures_info = parse_textures(spec, folders)

    materials: dict[str, RenderMaterial] = {}
    for mat_spec in spec.materials:
        assert isinstance(mat_spec, mj.MjsMaterial)
        if mat_spec.name in materials:
            print(f"[WARN]: material with name {mat_spec.name} already parsed")
            continue

        rgba = mat_spec.rgba.copy()
        em = mat_spec.emission
        emission_arr = [
            rgba[0].item() * em,
            rgba[1].item() * em,
            rgba[2].item() * em,
            1,
        ]
        render_material = RenderMaterial(
            emission=emission_arr,
            base_color=rgba.tolist(),
            specular=mat_spec.specular,
            roughness=1.0 - mat_spec.reflectance,
            metallic=mat_spec.shininess,
        )

        texture: RenderTexture2D | None = None
        texture_id = mat_spec.textures[mj.mjtTextureRole.mjTEXROLE_RGB]
        if texture_id and texture_id in textures_info:
            texture_filepath = textures_info[texture_id].file
            if texture_filepath.is_file():
                texture = RenderTexture2D(
                    filename=texture_filepath.as_posix(),
                    address_mode="repeat",
                    srgb=True,
                )

        if texture is not None:
            render_material.base_color_texture = texture
        materials[mat_spec.name] = render_material

    return materials


def has_any_non_free_joint(root_body: mj.MjsBody) -> bool:
    has_non_free_joint = False
    stack = [root_body]
    while stack:
        body = stack.pop()
        if body.joints and any(
            jnt.type != mj.mjtJoint.mjJNT_FREE for jnt in body.joints
        ):
            has_non_free_joint = True
            break
        stack.extend(body.bodies)

    return has_non_free_joint


def is_asset_articulated(filepath: Path) -> bool:
    spec = mj.MjSpec.from_file(filepath.as_posix())
    return any(jnt.type != mj.mjtJoint.mjJNT_FREE for jnt in spec.joints)


def visit_model_xml(filepath: Path, folders: MjcfAssetsFolders) -> None:
    root = ET.parse(filepath).getroot()

    meshdir, texturedir = filepath.parent, filepath.parent
    if (compiler_elem := root.find("compiler")) is not None:
        meshdir = filepath.parent / compiler_elem.get("meshdir", "")
        texturedir = filepath.parent / compiler_elem.get("texturedir", "")

    for mesh_elem in root.findall(".//asset/mesh"):
        if mesh_file_str := mesh_elem.get("file"):
            mesh_file = meshdir / mesh_file_str
            mesh_name = mesh_elem.get("name", mesh_file.stem)
            folders.meshes[mesh_name] = mesh_file

    for texture_elem in root.findall(".//asset/texture"):
        if texture_file_str := texture_elem.get("file"):
            texture_file = texturedir / texture_file_str
            texture_name = texture_elem.get("name", texture_file.stem)
            folders.textures[texture_name] = texture_file

    # TODO(wilbert): recurse along other sub models


def parse_xml(filepath: Path) -> tuple[mj.MjSpec, MjcfAssetsFolders]:
    spec = mj.MjSpec.from_file(filepath.as_posix())
    folders = MjcfAssetsFolders()

    root = ET.parse(filepath).getroot()

    for model_elem in root.findall(".//asset/model"):
        if model_path_str := model_elem.get("file"):
            model_filepath = filepath.parent / model_path_str
            visit_model_xml(model_filepath, folders)

    return spec, folders
