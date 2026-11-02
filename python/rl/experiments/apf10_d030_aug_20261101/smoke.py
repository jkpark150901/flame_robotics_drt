"""Validate split coverage, per-scene box augmentation, cleanup and reproducibility before training."""
import collections
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'python'))
from rl.path_env import PathPlanningEnv

cfg = json.loads((HERE / 'train.json').read_text())
env = PathPlanningEnv(cfg, seed=20261101)
split = json.loads((HERE / 'split.json').read_text())
actual = {(Path(t['snapshot']).name, t['group']) for t in env.targets}
assert actual == {tuple(t) for t in split['train']}, (len(actual), len(split['train']))
assert not actual & {tuple(t) for t in split['eval']}
env.rand_cfg['prob'] = 1.0
groups = {}
for target in env.targets:
    groups.setdefault((Path(target['snapshot']).name, target['phase']), target)
results = []
for index, (key, target) in enumerate(sorted(groups.items())):
    count = 0
    for attempt in range(6):
        seed = 510000 + index * 100 + attempt
        obs, info = env.reset(seed=seed, options={'target': target})
        assert np.isfinite(obs).all() and env.observation_space.contains(obs)
        assert len(env._boxes) == info['n_boxes'] == len(env._box_specs)
        specs = list(env._box_specs.values())
        if specs:
            count = len(specs)
            # Exact engine must see augmentation too, not just the reward SDF.
            sc = env.backend._handle(env.robot).pb_scene
            assert all(body in sc.static_bodies for body, _ in env._boxes)
            names = {sc.static_names[sc.static_bodies.index(body)] for body, _ in env._boxes}
            entries = env.backend.link_obstacle_distances(env.robot, env.q)
            assert names <= {e['obstacle'] for e in entries}
            body, sdf = env._boxes[0]
            env._remove_box((body, sdf))
            assert all(b != body for b, _ in env._boxes)
            assert body not in env._box_specs and body not in sc.static_bodies
            obs2, _ = env.reset(seed=seed, options={'target': target})
            assert specs == list(env._box_specs.values()) and np.allclose(obs, obs2)
            _, reward, _, _, _ = env.step(np.zeros(env.n))
            assert np.isfinite(reward)
            break
    assert count > 0, f'No accepted augmentation for {key}'
    results.append({'snapshot': key[0], 'phase': key[1], 'n_boxes': count, 'seed': seed})
    print('AUGMENTATION_OK', results[-1], flush=True)
env._clear_boxes()
assert not env._boxes and not env._box_specs
eval_cfg = json.loads((HERE / 'eval_clean.json').read_text())
evaluation = PathPlanningEnv(eval_cfg, seed=810000)
actual_eval = {(Path(t['snapshot']).name, t['group']) for t in evaluation.targets}
assert actual_eval == {tuple(t) for t in split['eval']}
print('SPLIT_OK', len(actual), len(actual_eval), flush=True)
(HERE / 'smoke_result.json').write_text(json.dumps({'train': len(actual), 'eval': len(actual_eval),
                                                 'augmentation': results}, indent=2) + '\n')
