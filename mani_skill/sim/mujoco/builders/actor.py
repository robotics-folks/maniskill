from __future__ import annotations

from pathlib import Path

import mujoco as mj
import numpy as np

from mani_skill.sim.core.builders import (
    BoxCollisionRecord,
    BoxVisualRecord,
    CapsuleCollisionRecord,
    CapsuleVisualRecord,
    ConvexMeshCollisionRecord,
    CylinderCollisionRecord,
    CylinderVisualRecord,
    MeshVisualRecord,
    MultipleConvexMeshCollisionRecord,
    NonconvexMeshCollisionRecord,
    PlaneCollisionRecord,
    PlaneVisualRecord,
    SphereCollisionRecord,
    SphereVisualRecord,
)

from ...core.builders.actor import ActorBuilder

DEFAULT_COLLIDER_RGBA = (0.8, 0.1, 0.1, 0.5)
DEFAULT_VISUAL_RGBA = (0.2, 0.2, 0.8, 1.0)

DEF_MUJOCO_RGBA = (0.5, 0.5, 0.5, 1.0)


def add_collision_records(
    scene_spec: mj.MjSpec,
    builder: ActorBuilder,
    body_spec: mj.MjsBody | None = None,
) -> None:
    body_spec = scene_spec.worldbody if body_spec is None else body_spec

    for col_rec in builder.collision_records:
        col_geom_spec: mj.MjsGeom | None = None
        match col_rec:
            case col if isinstance(col, PlaneCollisionRecord):
                col_geom_spec = body_spec.add_geom(
                    name=col.name,
                    type=mj.mjtGeom.mjGEOM_PLANE,
                    size=(10.0, 10.0, 0.01),
                )
            case col if isinstance(col, BoxCollisionRecord):
                col_geom_spec = body_spec.add_geom(
                    name=col.name,
                    type=mj.mjtGeom.mjGEOM_BOX,
                    size=col.half_size,
                )
            case col if isinstance(col, CapsuleCollisionRecord):
                col_geom_spec = body_spec.add_geom(
                    name=col.name,
                    type=mj.mjtGeom.mjGEOM_CAPSULE,
                    size=(col.radius, col.half_length, col.radius),
                )
            case col if isinstance(col, CylinderCollisionRecord):
                col_geom_spec = body_spec.add_geom(
                    name=col.name,
                    type=mj.mjtGeom.mjGEOM_CYLINDER,
                    size=(col.radius, col.half_length, col.radius),
                )
            case col if isinstance(col, SphereCollisionRecord):
                col_geom_spec = body_spec.add_geom(
                    name=col.name,
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    size=(col.radius, col.radius, col.radius),
                )
            case col if isinstance(col, ConvexMeshCollisionRecord):
                mesh_filepath = Path(col.filename)
                if not mesh_filepath.is_file():
                    print(
                        f"[ERROR]: convex mesh '{col.name}' points to invalid file @ '{col.filename}'"
                    )
                    continue

                mesh_name = f"{col.name}_collider"
                _ = scene_spec.add_mesh(
                    name=mesh_name,
                    file=mesh_filepath.as_posix(),
                    scale=col.scale,
                )

                col_geom_spec = body_spec.add_geom(
                    name=col.name,
                    type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=mesh_name,
                )
            case col if isinstance(col, MultipleConvexMeshCollisionRecord):
                print(
                    f"[WARNING]: Multiple convex colliders are not supported yet. Skipping collider '{col.name}'"
                )
            case col if isinstance(col, NonconvexMeshCollisionRecord):
                print(
                    f"[WARNING]: Non convex mesh colliders are not supported by MuJoCo. Skipping this collider '{col.name}'"
                )
            case _:
                print(
                    f"[WARNING]: collision record '{col.__class__.name}' not supported yet for collider '{col.name}'"
                )

        if col_geom_spec:
            col_geom_spec.rgba = np.array(col_rec.rgba)
            if col_geom_spec.type != mj.mjtGeom.mjGEOM_PLANE:
                col_geom_spec.density = col_rec.density


def add_visual_records(
    scene_spec: mj.MjSpec,
    builder: ActorBuilder,
    body_spec: mj.MjsBody | None = None,
) -> None:
    body_spec = scene_spec.worldbody if body_spec is None else body_spec

    for vis_rec in builder.visual_records:
        vis_is_rgba_default = np.allclose(vis_rec.material.rgba, DEF_MUJOCO_RGBA)
        vis_has_any_texture = vis_rec.material.has_any_texture()

        mat_name: str | None = None
        if not vis_is_rgba_default and vis_has_any_texture:
            mat_name = f"mat_{vis_rec.name}"
            scene_spec.add_material(
                name=mat_name,
                rgba=vis_rec.material.rgba,
            )

        vis_geom_spec: mj.MjsGeom | None = None
        match vis_rec:
            case vis if isinstance(vis, PlaneVisualRecord):
                vis_geom_spec = body_spec.add_geom(
                    name=vis.name,
                    type=mj.mjtGeom.mjGEOM_PLANE,
                    size=(10.0, 10.0, 0.01),
                )
            case vis if isinstance(vis, BoxVisualRecord):
                vis_geom_spec = body_spec.add_geom(
                    name=vis.name,
                    type=mj.mjtGeom.mjGEOM_BOX,
                    size=vis.half_size,
                )
            case vis if isinstance(vis, CapsuleVisualRecord):
                vis_geom_spec = body_spec.add_geom(
                    name=vis.name,
                    type=mj.mjtGeom.mjGEOM_CAPSULE,
                    size=(vis.radius, vis.half_length, vis.radius),
                )
            case vis if isinstance(vis, CylinderVisualRecord):
                vis_geom_spec = body_spec.add_geom(
                    name=vis.name,
                    type=mj.mjtGeom.mjGEOM_CYLINDER,
                    size=(vis.radius, vis.half_length, vis.radius),
                )
            case vis if isinstance(vis, SphereVisualRecord):
                vis_geom_spec = body_spec.add_geom(
                    name=vis.name,
                    type=mj.mjtGeom.mjGEOM_SPHERE,
                    size=(vis.radius, vis.radius, vis.radius),
                )
                pass
            case vis if isinstance(vis, MeshVisualRecord):
                mesh_filepath = Path(vis.filename)
                if not mesh_filepath.is_file():
                    print(
                        f"[ERROR]: convex mesh '{vis.name}' points to invalid file @ '{vis.filename}'"
                    )
                    continue

                mesh_name = f"{vis.name}_collider"
                _ = scene_spec.add_mesh(
                    name=mesh_name,
                    file=mesh_filepath.as_posix(),
                    scale=vis.scale,
                )

                vis_geom_spec = body_spec.add_geom(
                    name=vis.name,
                    type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=mesh_name,
                )

        if vis_geom_spec:
            vis_geom_spec.contype = 0
            vis_geom_spec.conaffinity = 0

            if mat_name:
                vis_geom_spec.material = mat_name
            else:
                vis_geom_spec.rgba = np.array(vis.material.rgba)
