from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np
import trimesh
import tyro
from tqdm import tqdm

MJCF_DIR = Path(__file__).parent.parent.parent / "assets-molmospaces" / "mjcf"
MJCF_OBJECTS_DIR = MJCF_DIR / "objects"
MJCF_SCENES_DIR = MJCF_DIR / "scenes"

MJCF_ASSETS_THOR_DIR = MJCF_OBJECTS_DIR / "thor"
MJCF_SCENES_ITHOR_DIR = MJCF_SCENES_DIR / "ithor"

COLLIDERS_CLASSNAMES = (
    "__DYNAMIC_MJT__",
    "__STRUCTURAL_MJT__",
    "__ARTICULABLE_DYNAMIC_MJT__",
)

SCENE_DATASET_IDS = (
    "procthor-10k-train",
    "procthor-10k-val",
    "procthor-10k-test",
    "procthor-objaverse-train",
    "procthor-objaverse-val",
    "holodeck-objaverse-train",
    "holodeck-objaverse-val",
)


@dataclass
class Args:
    thor_assets_dir: Path = MJCF_ASSETS_THOR_DIR
    scenes_ithor_dir: Path = MJCF_SCENES_ITHOR_DIR

    fix_skybox_layout: bool = False


def resize_thor_asset(model_path: Path) -> None:
    modified = False
    try:
        spec = mj.MjSpec.from_file(model_path.as_posix())

        for geom in spec.geoms:
            assert isinstance(geom, mj.MjsGeom)
            if geom.type != mj.mjtGeom.mjGEOM_MESH:
                continue
            if not geom.classname.name in COLLIDERS_CLASSNAMES:
                continue
            mesh_handle = spec.mesh(geom.meshname)
            if mesh_handle is None:
                continue

            mesh_path = model_path.parent / mesh_handle.file
            if not mesh_path.is_file():
                continue

            scale = mesh_handle.scale
            if np.allclose(scale, np.ones_like(scale), atol=1e-3):
                continue
            modified = True

            mesh = trimesh.load_mesh(mesh_path.as_posix())
            if isinstance(mesh, list):
                mesh = mesh[0]
            mesh.apply_scale(mesh_handle.scale)

            with open(mesh_path.resolve(), "w") as fhandle:
                trimesh.exchange.export.export_mesh(  # pyright: ignore [reportAttributeAccessIssue]
                    mesh,
                    fhandle,
                    file_type="obj",
                )

            mesh_handle.scale = np.ones_like(scale)

        if modified:
            _ = spec.compile()
            with open(model_path.resolve(), "w") as fhandle:
                fhandle.write(spec.to_xml())

    except Exception as e:  # noqa: BLE001
        print(f"[ERROR]: couldn't resize thor assets '{model_path.stem}', error: {e}")


def update_scenes_scales(scene_path: Path) -> None:
    modified = False
    try:
        spec = mj.MjSpec.from_file(scene_path.as_posix())
        for geom in spec.geoms:
            assert isinstance(geom, mj.MjsGeom)
            if geom.type != mj.mjtGeom.mjGEOM_MESH:
                continue
            if not geom.classname.name in COLLIDERS_CLASSNAMES:
                continue
            mesh_handle = spec.mesh(geom.meshname)
            if mesh_handle is None:
                continue

            if not "objects/thor" in mesh_handle.file:
                continue

            scale = mesh_handle.scale
            if np.allclose(scale, np.ones_like(scale), atol=1e-3):
                continue
            modified = True
            mesh_handle.scale = np.ones_like(scale)

        if modified:
            _ = spec.compile()
            with open(scene_path.resolve(), "w") as fhandle:
                fhandle.write(spec.to_xml())

        patch_scene_xml(scene_path)

    except Exception as e:  # noqa: BLE001
        print(f"[ERROR]: couldn't resize scene '{scene_path.stem}', error: {e}")


def patch_scene_xml(scene_path: Path) -> None:
    # Skyboxes in mjspec are still broken when parsed T_T, have to patch the xml file directly
    tree = ET.parse(scene_path)
    root = tree.getroot()

    if asset_elm := root.find("asset"):
        for texture_elm in asset_elm.findall("texture"):
            if texture_elm.attrib.get("type", "2d") == "skybox":
                texture_elm.attrib["gridlayout"] = "LFRB.D.."

    tree.write(scene_path)


def main() -> int:
    args = tyro.cli(Args)

    if not args.thor_assets_dir.is_dir():
        print(f"thor assets dir @ {args.thor_assets_dir} is not a valid directory")
        return 1

    if not args.scenes_ithor_dir.is_dir():
        print(f"scenes ithor dir @ {args.scenes_ithor_dir} is not a valid directory")
        return 1

    models_filepaths: list[Path] = []
    for candidate_xml in args.thor_assets_dir.rglob("*.xml"):
        if any(
            substr in candidate_xml.stem
            for substr in ("_old", "_fix", "_upt", "_orig", "_sc")
        ):
            continue
        if "_mesh" not in candidate_xml.stem:
            continue
        models_filepaths.append(candidate_xml)

    for model_path in tqdm(models_filepaths):
        resize_thor_asset(model_path)

    scenes_filepaths: list[Path] = []
    for candidate_xml in args.scenes_ithor_dir.glob("*.xml"):
        if any(
            substr in candidate_xml.stem
            for substr in ("_old", "_fix", "_upt", "_orig", "_sc")
        ):
            continue
        scenes_filepaths.append(candidate_xml)

    for dataset_id in SCENE_DATASET_IDS:
        dataset_folder = MJCF_SCENES_DIR / dataset_id
        if not dataset_folder.is_dir():
            continue
        scenes_filepaths.extend(dataset_folder.rglob("*.xml"))

    for scene_path in tqdm(scenes_filepaths):
        if args.fix_skybox_layout:
            patch_scene_xml(scene_path)
        else:
            update_scenes_scales(scene_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
