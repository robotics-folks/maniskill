import os
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import trimesh
import tyro


@dataclass
class Args:
    filepath: Path


def main() -> int:
    args = tyro.cli(Args)

    if not args.filepath.is_file():
        return 1

    spec = mj.MjSpec.from_file(args.filepath.as_posix())

    for mesh_spec in spec.meshes:
        if all(suffix not in mesh_spec.file for suffix in (".stl", ".STL")):
            continue
        mesh_path = args.filepath.parent / spec.meshdir / mesh_spec.file
        if not mesh_path.is_file():
            raise FileNotFoundError(f"[ERROR]: the mesh @ {mesh_path} doesn't exist")

        mesh = trimesh.load_mesh(mesh_path.as_posix())
        if isinstance(mesh, list):
            mesh = mesh[0]

        mesh_path_obj = (
            args.filepath.resolve().parent / spec.meshdir / f"{mesh_path.stem}.obj"
        )
        mesh_path_obj_link = args.filepath.parent / spec.meshdir / mesh_path_obj.name

        with open(mesh_path_obj.resolve(), "w") as fhandle:
            trimesh.exchange.export.export_mesh(  # pyright: ignore [reportAttributeAccessIssue]
                mesh,
                fhandle,
                file_type="obj",
            )

        if mesh_path_obj_link.exists():
            mesh_path_obj_link.unlink(missing_ok=True)
        mesh_path_obj_link.symlink_to(mesh_path_obj)
        mesh_spec.file = os.path.join(
            os.path.dirname(mesh_spec.file), mesh_path_obj.name
        )

    _ = spec.compile()
    with open(args.filepath, "w") as fhandle:
        fhandle.write(spec.to_xml())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
