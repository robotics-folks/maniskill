from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np
import trimesh
import tyro
from p_tqdm import p_uimap
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


@dataclass
class Args:
    objects: list[Literal["thor", "objaverse"]] = field(default_factory=list)
    scenes: list[
        Literal[
            "ithor",
            "procthor-10k-train",
            "procthor-10k-val",
            "procthor-10k-test",
            "procthor-objaverse-train",
            "procthor-objaverse-val",
            "holodeck-objaverse-train",
            "holodeck-objaverse-val",
        ]
    ] = field(default_factory=list)

    num_workers: int = 1


def resize_thor_asset(model_path: Path) -> tuple[bool, str]:
    success = True
    error_msg = ""
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

            scale = mesh_handle.scale
            if np.allclose(scale, np.ones_like(scale), atol=1e-3):
                continue

            mesh_path = model_path.parent / mesh_handle.file
            if not mesh_path.is_file():
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
        success = False
        error_msg = (
            f"[ERROR]: couldn't resize thor assets '{model_path.stem}', error: {e}"
        )

    return success, error_msg


def update_scenes_scales(scene_path: Path) -> tuple[bool, bool, str]:
    success = True
    error_msg = ""
    modified = False

    try:
        tree = ET.parse(scene_path)
        root = tree.getroot()

        mesh_ids: set[str] = set()
        for geom_elm in root.findall(".//geom"):
            if geom_elm.get("class", "") not in COLLIDERS_CLASSNAMES:
                continue
            if mesh_id := geom_elm.get("mesh"):
                mesh_ids.add(mesh_id)

        for mesh_elm in root.findall(".//asset/mesh"):
            if not mesh_elm.get("name", "") in mesh_ids:
                continue
            if scale_str := mesh_elm.get("scale"):
                scale = np.array(tuple(map(float, scale_str.split(" "))))
                if np.allclose(scale, np.ones_like(scale), atol=1e-3):
                    continue
                modified = True
                mesh_elm.set("scale", "1 1 1")

        if modified:
            tree.write(scene_path.resolve())

    except Exception as e:  # noqa: BLE001
        success = False
        error_msg = f"[ERROR]: couldn't resize scene '{scene_path.stem}', error: {e}"

    return success, modified, error_msg


def patch_scene_xml(scene_path: Path) -> None:
    # Skyboxes in mjspec are still broken when parsed T_T, have to patch the xml file directly
    tree = ET.parse(scene_path)
    root = tree.getroot()

    if asset_elm := root.find("asset"):
        for texture_elm in asset_elm.findall("texture"):
            if texture_elm.attrib.get("type", "2d") == "skybox":
                texture_elm.attrib["gridlayout"] = "LFRB.D.."

    tree.write(scene_path.resolve())


# TODO(wilbert): Seems that xml ElementTree doens't play well with tqdm, so the progress bar is
# messed up T_T. Might have to try another library like lxml or find a workaround for this
def main() -> int:
    args = tyro.cli(Args)

    for dataset_id in args.objects:
        objects_dir = MJCF_OBJECTS_DIR / dataset_id
        if not objects_dir.is_dir():
            continue

        models_filepaths: list[Path] = []
        for candidate_xml in objects_dir.rglob("*.xml"):
            if any(
                substr in candidate_xml.stem
                for substr in ("_old", "_fix", "_upt", "_orig", "_sc")
            ):
                continue
            if "_mesh" not in candidate_xml.stem:
                continue
            models_filepaths.append(candidate_xml)

        print(f"Resizing Objects Dataset: '{dataset_id}'")
        if args.num_workers > 1:
            results = p_uimap(
                resize_thor_asset, models_filepaths, num_cpus=args.num_workers
            )
            for success, error_msg in results:
                if not success:
                    print(error_msg)

        else:
            for model_path in tqdm(models_filepaths):
                success, error_msg = resize_thor_asset(model_path)
                if not success:
                    print(error_msg)

    for dataset_id in args.scenes:
        dataset_folder = MJCF_SCENES_DIR / dataset_id
        if not dataset_folder.is_dir():
            continue

        scenes_filepaths = list(
            dataset_folder.rglob("*.xml")
            if dataset_id != "ithor"
            else dataset_folder.glob("*.xml")
        )

        print(f"Resizing Scenes Dataset: '{dataset_id}'")

        num_total = 0
        num_modified = 0
        if args.num_workers > 1:
            results = p_uimap(
                update_scenes_scales, scenes_filepaths, num_cpus=args.num_workers
            )
            for success, modified, error_msg in results:
                num_total += 1
                if success and modified:
                    num_modified += 1
                else:
                    print(error_msg)
        else:
            for scene_path in tqdm(scenes_filepaths):
                success, modified, error_msg = update_scenes_scales(scene_path)
                num_total += 1
                if success and modified:
                    num_modified += 1
                else:
                    print(error_msg)

        print(f"Number of scenes modified: {num_modified} / {num_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
