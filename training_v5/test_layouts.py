"""Generated construction layouts are connected, clear of the agents and varied."""
import collections
import unittest

import numpy as np

from physics import PhysicsEnv, generate_arena, local_xy

SCENARIOS = ['connected-rooms', 'corridors', 'multi-exit']
GRID_STEP = .15
AGENT_CLEARANCE = .27
SPAWN_CLEARANCE = .25


def wall_distance(point, wall):
    """Distance from a point to the footprint of an axis-aligned-in-its-frame wall or object."""
    u, v = local_xy(point, wall['yaw'])
    return np.hypot(max(0, abs(u) - wall['size'][0] / 2), max(0, abs(v) - wall['size'][1] / 2))


def free_cells(arena):
    coords = np.arange(.3, arena['width'] - .29, GRID_STEP)
    free = set()
    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            if all(wall_distance([x - w['position'][0], y - w['position'][1]], w) >= AGENT_CLEARANCE for w in arena['walls']):
                free.add((i, j))
    return free


def reachable(free):
    seen = {next(iter(free))}
    queue = collections.deque(seen)
    while queue:
        i, j = queue.popleft()
        for point in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            if point in free and point not in seen:
                seen.add(point)
                queue.append(point)
    return seen


class Layouts(unittest.TestCase):
    def test_connected_clearance_and_variety(self):
        fingerprints = set()
        for scenario in SCENARIOS:
            for seed in range(40):
                a = generate_arena(seed + 901, scenario, size=8 + seed % 5, n_boxes=6, n_ramps=2)
                self.assertLessEqual(len(a['walls']), 16)
                free = free_cells(a)
                self.assertEqual(len(reachable(free)), len(free), (scenario, seed))
                for agent in a['agents']:
                    for w in a['walls'] + a['objects']:
                        self.assertGreaterEqual(wall_distance(np.array(agent['position']) - w['position'], w), SPAWN_CLEARANCE)
                fingerprints.add(str(a['walls']))
            for seed in range(3):
                e = PhysicsEnv(seed=700 + seed, scenario=scenario, size=10, n_boxes=6, n_ramps=2)
                for _ in range(240):
                    obs, reward, done, info = e.step(np.array([[.4, .2, .1, 0, 0, 0], [-.3, .1, .1, 0, 0, 0]]))
                self.assertTrue(done)
                self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(len(fingerprints), 120)


if __name__ == '__main__':
    unittest.main()
