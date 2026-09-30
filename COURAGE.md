# Some info about the changes for the COURAGE project

## What is this ?

I'm currently working on (`wpumacay`) adding any features I would need down the line in the core
of `ManiSkill` for a separate project called `Courage`, which started as a reimplementation of
`MolmoSpaces` but with support for many physics backend, rendering backends, visualizers, tooling,
etc. . The main repos and the forks are on codeberg, but the ManiSkill repo is quite a chunky one,
(like, we shouldn't have all the assets in the repo :sweat_smile:, or at least use huggingface and
a resources manager for all) and bc of the storage quota limit on codeberg I had to move back this
fork github :tear_smile:.

## How to use

### Installation

Make sure you install the `courage` dep group when installing ManiSkill from this fork, like this:

```bash
uv pip install -e .[dev,courage] # if in macOS, add quotes like this: ".[dev,courage]"
```

This will install `mujoco` (required for using `MjSpec` to handle `mjcf` models), `molmospaces-resources`
from a separate fork I'm using for a rewrite (will need this to download the `MolmoSpaces` assets),
and `viser` (I'm currently making a viser viewer for this project, based on [this](https://github.com/mani-skill/ManiSkill/pull/1466) PR, but just as reference
as I had a separate working implementation for `mujoco`, `isaacsim` and `maniskill` that had been keeping dust for
some months already :sweat_smile:. If you want just to use `mujoco`, Kevin Zakka's [mjviser](https://github.com/mujocolab/mjviser) is what you want)

### Download the assets

I have some extra functionality in the resources manager fork that you can use to download the assets
in one go and specifying what you want, like shown below:

```bash
# Download assets in batch mode, download thor objects, ithor and procthor-10k-val scenes, and some robots
molmospaces-download --type mjcf --mode batch --batch.objects thor --batch.scenes ithor procthor-10k-val --batch.robots franka_droid i2rt_yam g1
```

For more info on this script just run:

```bash
molmospaces-download --help
```

Once you have the assets, you can use the following script to convert the meshes to scale 1, bc it
seems PhysX doesn't play well with scaled mesh colliders :tear_smile:, similar to `IsaacSim` in versions
previous to 6.x :

```bash
python scripts/courage/resize_mjcf_assets_and_scenes.py --scale-thor --scale-ithor --num-workers 10
```

This will re-scale the assets and the houses as well, for all `thor` assets, and `ithor` houses. I'll
modify the script to handle the other houses dataset, sorry for the delay :sweat_smile: .

For some robots, it seems that they have `stl` meshes for both colliders and visuals (e.g. `franka_droid` gripper),
so we have to run another script to convert those meshes to `obj` and update the mjcf model file:

```bash
python scripts/courage/make_obj_visuals_from_stl.py --filepath assets-molmospaces/mjcf/robots/franka_droid/robotiq_2f85_v4/2f85.xml
```

### Use the houses (IN CONSTRUCTION)

Will add more docs on this part, but for now you can try out the env wrapper made for the molmospaces
houses and assets, like this:

```bash
python -m mani_skill.examples.demo_mjcf_loader --scene-path assets-molmospaces/mjcf/scenes/ithor/FloorPlan1_physics.xml --render-mode human
```
