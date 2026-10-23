# Defines typed geometry and joint records shared by backend-agnostic builders.

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from PIL import Image

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
JointLimits = tuple[float, float]

Color4 = tuple[float, float, float, float]

UNBOUNDED_LIMITS: JointLimits = (float("-inf"), float("inf"))

DEFAULT_VISUAL_RGBA = (1.0, 1.0, 1.0, 1.0)
DEFAULT_COLLIDER_RGBA = (0.1, 0.1, 0.8, 1.0)

NP_MAX_ARR_PRINT_SIZE = 10


@dataclass(frozen=True, slots=True, kw_only=True)
class PoseRecord:
    """Describe a backend-agnostic rigid transform."""

    position: Vector3 = (0.0, 0.0, 0.0)
    """XYZ position in 3d space"""

    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0)
    """XYZW quaternion (scalar-last) representing the orientation in 3d space"""


# -------------------------------------------------------------------------------------------------
# Collisions-related structs
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _CollisionRecordBase:
    """Provide fields shared by every typed collision record."""

    name: str = ""
    pose: PoseRecord = PoseRecord()
    material: object | None = None
    density: float = 1000.0


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaneCollisionRecord(_CollisionRecordBase):
    """Describe an infinite plane collision."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BoxCollisionRecord(_CollisionRecordBase):
    """Describe a box collision."""

    half_size: Vector3 = (1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class CapsuleCollisionRecord(_CollisionRecordBase):
    """Describe a capsule collision."""

    radius: float = 1.0
    half_length: float = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class CylinderCollisionRecord(_CollisionRecordBase):
    """Describe a cylinder collision."""

    radius: float = 1.0
    half_length: float = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SphereCollisionRecord(_CollisionRecordBase):
    """Describe a sphere collision."""

    radius: float = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvexMeshCollisionRecord(_CollisionRecordBase):
    """Describe one convex collision mesh loaded from a file."""

    filename: str
    scale: Vector3 = (1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class MultipleConvexMeshCollisionRecord(_CollisionRecordBase):
    """Describe multiple convex collision meshes loaded from a file."""

    filename: str
    scale: Vector3 = (1.0, 1.0, 1.0)
    decomposition: Literal["none", "coacd"] = "none"
    decomposition_params: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class NonconvexMeshCollisionRecord(_CollisionRecordBase):
    """Describe one non-convex collision mesh loaded from a file."""

    filename: str
    scale: Vector3 = (1.0, 1.0, 1.0)


CollisionRecord: TypeAlias = (
    PlaneCollisionRecord
    | BoxCollisionRecord
    | CapsuleCollisionRecord
    | CylinderCollisionRecord
    | SphereCollisionRecord
    | ConvexMeshCollisionRecord
    | MultipleConvexMeshCollisionRecord
    | NonconvexMeshCollisionRecord
)

# -------------------------------------------------------------------------------------------------
# Visuals-related structs
# -------------------------------------------------------------------------------------------------


def custom_str_repr(obj: Any) -> str:
    r_str = f"<{obj.__class__.__name__}\n"
    for f_name, f_value in obj.__dict__.items():
        if f_value is None:
            r_str += f"  {f_name}: None\n"
        elif isinstance(f_value, (str, int, float, bool)):
            r_str += f"  {f_name}: {f_value}\n"
        elif isinstance(f_value, Image.Image):
            r_str += f"  {f_name}: (width={f_value.width}, height={f_value.height})\n"
        else:
            r_str += f"  {f_name}: {repr(f_value)}"
    r_str += ">\n"
    return r_str


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterialRecord:
    name: str = ""
    """The unique identifier associated with this material"""

    rgba: Color4 = field(default_factory=lambda: copy.copy(DEFAULT_VISUAL_RGBA))
    """RGB+Alpha color of this material"""

    tex_albedo: Image.Image | None = None
    """(Optional) The texture representing the albedo map"""

    tex_normal: Image.Image | None = None
    """(Optional) The texture representing the normal map"""

    tex_roughness: Image.Image | None = None
    """(Optional) The texture representing the roughness map"""

    def has_any_texture(self) -> bool:
        return (
            self.tex_albedo is not None
            or self.tex_normal is not None
            or self.tex_roughness is not None
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _VisualRecordBase:
    """Provide fields shared by every typed visual record."""

    name: str = ""
    """Unique identifier of this visual record"""

    pose: PoseRecord = field(default_factory=PoseRecord)
    """Pose of this visual relative to its owner (body, link, etc.)"""

    material: MaterialRecord = field(default_factory=MaterialRecord)
    """The material struct containing the visual properties of this record"""

    def __repr__(self) -> str:
        return custom_str_repr(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaneVisualRecord(_VisualRecordBase):
    """Describe a plane visual."""

    width: float = 10.0
    """Width of the plane (x-axis)"""

    length: float = 10.0
    """Length of the plane (y-axis)"""


@dataclass(frozen=True, slots=True, kw_only=True)
class BoxVisualRecord(_VisualRecordBase):
    """Describe a box visual."""

    half_size: Vector3 = (1.0, 1.0, 1.0)
    """Half extents of the box along x | y | z axes"""


@dataclass(frozen=True, slots=True, kw_only=True)
class CapsuleVisualRecord(_VisualRecordBase):
    """Describe a capsule visual."""

    radius: float = 1.0
    """Radius of both top-bottom spherical caps"""

    half_length: float = 1.0
    """Half length of the cylindrical part"""


@dataclass(frozen=True, slots=True, kw_only=True)
class CylinderVisualRecord(_VisualRecordBase):
    """Describe a cylinder visual."""

    radius: float = 1.0
    """Radius of both top-bottom circular sections"""

    half_length: float = 1.0
    """Half length of the whole cylinder (top-bottom)"""


@dataclass(frozen=True, slots=True, kw_only=True)
class SphereVisualRecord(_VisualRecordBase):
    """Describe a sphere visual."""

    radius: float = 1.0
    """Radius of the sphere"""


@dataclass(frozen=True, slots=True, kw_only=True)
class MeshVisualRecord(_VisualRecordBase):
    """Describe a visual mesh loaded from a file."""

    filename: str
    """Path to the file used for this mesh"""

    scale: Vector3 = (1.0, 1.0, 1.0)
    """Scale of the mesh"""


VisualRecord: TypeAlias = (
    PlaneVisualRecord
    | BoxVisualRecord
    | CapsuleVisualRecord
    | CylinderVisualRecord
    | SphereVisualRecord
    | MeshVisualRecord
)

# -------------------------------------------------------------------------------------------------
# Joints-related structs
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class _JointRecordBase:
    """Provide fields shared by every typed articulation joint record."""

    name: str = ""
    pose_in_parent: PoseRecord = PoseRecord()
    pose_in_child: PoseRecord = PoseRecord()
    friction: float = 0.0
    damping: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class UndefinedJointRecord(_JointRecordBase):
    """Describe an unspecified root joint."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FixedJointRecord(_JointRecordBase):
    """Describe a fixed joint."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RevoluteJointRecord(_JointRecordBase):
    """Describe a bounded revolute joint."""

    limits: JointLimits = UNBOUNDED_LIMITS


@dataclass(frozen=True, slots=True, kw_only=True)
class RevoluteUnwrappedJointRecord(_JointRecordBase):
    """Describe an unwrapped revolute joint."""

    limits: JointLimits = UNBOUNDED_LIMITS


@dataclass(frozen=True, slots=True, kw_only=True)
class PrismaticJointRecord(_JointRecordBase):
    """Describe a bounded prismatic joint."""

    limits: JointLimits = UNBOUNDED_LIMITS


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuousJointRecord(_JointRecordBase):
    """Describe a continuous revolute joint."""


JointRecord: TypeAlias = (
    UndefinedJointRecord
    | FixedJointRecord
    | RevoluteJointRecord
    | RevoluteUnwrappedJointRecord
    | PrismaticJointRecord
    | ContinuousJointRecord
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MimicJointRecord:
    """Describe a linear mimic relationship between two named joints."""

    joint: str
    mimic: str
    multiplier: float = 1.0
    offset: float = 0.0
