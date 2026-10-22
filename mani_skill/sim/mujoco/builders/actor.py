from __future__ import annotations

from pathlib import Path

import mujoco as mj

from mani_skill.sim.core.builders import (
    BoxCollisionRecord,
    CapsuleCollisionRecord,
    ConvexMeshCollisionRecord,
    CylinderCollisionRecord,
    MultipleConvexMeshCollisionRecord,
    NonconvexMeshCollisionRecord,
    PlaneCollisionRecord,
    SphereCollisionRecord,
)

from ...core.builders.actor import ActorBuilder

DEFAULT_COLLIDER_RGBA = (0.8, 0.1, 0.1, 0.5)
DEFAULT_VISUAL_RGBA = (0.2, 0.2, 0.8, 1.0)


def add_actor(
    scene_spec: mj.MjSpec,
    builder: ActorBuilder,
    build_physics: bool,
    build_render: bool,
) -> None:
    actor_rootbody = scene_spec.worldbody.add_body(
        name=builder.name,
        pos=builder.initial_pose.position,
        quat=builder.initial_pose.orientation,
    )

    if build_physics:
        for col_rec in builder.collision_records:
            match col_rec:
                case col if isinstance(col, PlaneCollisionRecord):
                    # TODO(wilbert): infinite planes in mujoco are only geoms of worldbody, right?
                    pass
                case col if isinstance(col, BoxCollisionRecord):
                    actor_rootbody.add_geom(
                        name=col.name,
                        type=mj.mjtGeom.mjGEOM_BOX,
                        size=col.half_size,
                        density=col.density,
                        rgba=DEFAULT_COLLIDER_RGBA,
                    )
                case col if isinstance(col, CapsuleCollisionRecord):
                    actor_rootbody.add_geom(
                        name=col.name,
                        type=mj.mjtGeom.mjGEOM_CAPSULE,
                        size=(col.radius, col.half_length, col.radius),
                        density=col.density,
                        rgba=DEFAULT_COLLIDER_RGBA,
                    )
                case col if isinstance(col, CylinderCollisionRecord):
                    actor_rootbody.add_geom(
                        name=col.name,
                        type=mj.mjtGeom.mjGEOM_CYLINDER,
                        size=(col.radius, col.half_length, col.radius),
                        density=col.density,
                        rgba=DEFAULT_COLLIDER_RGBA,
                    )
                case col if isinstance(col, SphereCollisionRecord):
                    actor_rootbody.add_geom(
                        name=col.name,
                        type=mj.mjtGeom.mjGEOM_SPHERE,
                        size=(col.radius, col.radius, col.radius),
                        density=col.density,
                        rgba=DEFAULT_COLLIDER_RGBA,
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

                    actor_rootbody.add_geom(
                        name=col.name,
                        type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=mesh_name,
                        density=col.density,
                        rgba=DEFAULT_COLLIDER_RGBA,
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

    # if build_render:
    #     for vis_rec in builder.visual_records:
    #         pass
