# For 4.7 (my own scene): drop a cloth on a spaceship
# uv run python main.py 4_6 scenes/hw4/cloth_spaceship.py
#
# Spaceship model: "craft_speederD" from Space Kit by Kenney (CC0),
# see scenes/hw4/spaceship/CREDITS.txt

import numpy as np
import trimesh
from pathlib import Path
from scenes.hw4.geometry import make_cloth_grid, make_cloth_constraints

SHIP_PARTS = ['ship_dark.ply', 'ship_metal.ply',
              'ship_metalDark.ply', 'ship_metalRed.ply']

def make_scene():
    # The ship is split into one file per material, but the
    # simulator only takes one collision mesh, so merge them.
    folder = Path(__file__).with_name('spaceship')
    vertices, normals, triangles = [], [], []
    offset = 0
    for name in SHIP_PARTS:
        mesh = trimesh.load_mesh(folder / name, process=False)
        vertices.append(np.asarray(mesh.vertices, dtype=np.float32))
        normals.append(np.asarray(mesh.vertex_normals, dtype=np.float32))
        triangles.append(np.asarray(mesh.faces, dtype=np.uint32) + offset)
        offset += len(mesh.vertices)
    vertices = np.concatenate(vertices)
    normals = np.concatenate(normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    triangles = np.concatenate(triangles)

    # Make the ship bigger and let it hover a bit above the floor
    vertices *= 2.0
    vertices[:, 1] += 0.6 - vertices[:, 1].min()
    ship_top = vertices[:, 1].max()

    # Smaller cloth, tilted and nudged sideways so it slides off a wing
    # 37x37 (same as the rock scene): a finer cloth jitters on the
    # ship's sharp cockpit edges since each point only gets one contact plane
    nx, ny = 37, 37
    sx, sy = 2.5, 2.5
    positions, cloth_triangles, dx, dy = make_cloth_grid(nx, ny, sx, sy)
    tilt = 0.3  # radians
    x = positions[:, 0].copy()
    positions[:, 0] = np.cos(tilt) * x + 0.4
    positions[:, 1] = ship_top + 0.7 + np.sin(tilt) * x
    edges, rest_lengths = make_cloth_constraints(nx, ny, dx, dy)

    velocities = np.zeros_like(positions)
    velocities[:, 0] = 0.5

    return {
        'positions': positions,
        'velocities': velocities,
        'inv_masses': np.ones(len(positions), dtype=np.float32),

        'constraint_edges': edges,
        'rest_lengths': rest_lengths,
        'triangles': cloth_triangles,
        'cloth_grid_size': (nx, ny),
        'spacing': (dx, dy),

        'collision_mesh': {
            'vertices': np.ascontiguousarray(vertices),
            'triangles': np.ascontiguousarray(triangles),
            'vertex_normals': np.ascontiguousarray(normals),
        },

        # Floor at y = 0, no spheres
        'planes': np.array([
            [0, 1, 0, 0],
        ], dtype=np.float32),
        'spheres': np.zeros((0, 4), dtype=np.float32),

        'collision_distance': 0.025,
        'contact_margin': 0.05,
        'static_friction': 0.55,
        'kinetic_friction': 0.35,

        'substeps': 6,
        'iterations': 20,

        'cam_pos': (3.5, 3.0, -3.5),
        'cam_target': (0.0, 0.8, 0.0),
        'cam_up': (0.0, 1.0, 0.0),
    }
