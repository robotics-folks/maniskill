# MuJoCo format files loaders

<!-- TODO(wilbert): give more information -->

## Download molmospaces mjcf assets

```bash
molmospaces-download --type mjcf --mode batch --batch.objects thor --batch.scenes ithor procthor-10k-test
```

## Process the assets to be usable by PhysX

There's an issue at the moment with scaling factors for colliders in `PhysX`. Seems that even though
the assets and scenes do use scaling factors for some assets, even if this scales are passed to the
actor builders, it breaks the simulation (similar to the issue we had back in the day with IsaacSim
and colliders with scaling factors bigger than 1).

To resolve this, a workaround is to pre-process the assets and scenes such that the mesh colliders
have unit scale, which makes it play well with the `PhysX` backend. After downloading the assets,
just run the `resize_mjcf_assets_and_scenes.py`:

```bash
python scripts/courage/resize_mjcf_assets_and_scenes.py
```

This will resize the mesh colliders using `trimesh` and update the associated `mjcf` files such that
it's usable by `PhysX`.
