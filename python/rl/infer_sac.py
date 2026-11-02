"""SAC inference on a saved snapshot/reference run, exporting standard SimTool playback.

Inference is deterministic; augmentation and train/eval filters are disabled.
The selected robot alone is exported (other robots are never labelled as SAC results).
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))


def inference_config(path, reference, shield, targets=None):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    cfg = copy.deepcopy(raw.get('cfg', raw))
    env = cfg['env']
    if not env.get('residual', {}).get('enabled'):
        raise ValueError('This interface requires a residual SAC configuration.')
    env.pop('target_split', None)
    env.pop('target_filter', None)
    if targets:
        env['target_filter'] = targets
    env.update(start_perturb_prob=0.0, curriculum={'enabled': False}, random_obstacles={'enabled': False})
    env['residual'].update(reference_runs={'*': str(Path(reference).resolve())},
                           random_reference=False, mid_start_prob=0.0, start_noise_prob=0.0, shield=shield)
    return cfg


def emit(kind, **values):
    print('SAC_UI ' + json.dumps({'type': kind, **values}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'config', 'snapshot', 'reference', 'out']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--shield', action='store_true')
    ap.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    ap.add_argument('--target', action='append', help='Exact target group name; default all eligible targets')
    args = ap.parse_args()
    for p in [args.model, args.config, args.snapshot, str(Path(args.reference) / 'summary.csv')]:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'Output folder already exists: {out}')
    cfg = inference_config(args.config, args.reference, args.shield, args.target)
    from rl.path_env import PathPlanningEnv
    from stable_baselines3 import SAC
    from plugins.robotics.plan_run_io import save_plan_run
    import numpy as np
    import torch
    torch.set_num_threads(2)
    emit('status', message='장면 및 학습 모델 로딩 중…')
    env = PathPlanningEnv(cfg, snapshots=[str(Path(args.snapshot).resolve())], seed=810000)
    model = SAC.load(args.model, device=args.device)
    if (model.observation_space.shape != env.observation_space.shape
            or model.action_space.shape != env.action_space.shape):
        raise ValueError('Model observation/action dimensions do not match the selected configuration and scene.')
    with open(Path(args.reference) / 'summary.csv', encoding='utf-8-sig') as f:
        ref_rows = [r for r in csv.DictReader(f) if r['robot_name'] == env.robot]
    by_group = {r['group_name']: r for r in ref_rows}
    eligible = {t['group'] for t in env.targets}
    skipped = [r['group_name'] for r in ref_rows if r['group_name'] not in eligible
               and (not args.target or r['group_name'] in args.target)]
    if args.target and set(args.target) - eligible:
        raise ValueError(f'No compatible stored goal/reference for: {sorted(set(args.target) - eligible)}')
    order = {r['group_name']: i for i, r in enumerate(ref_rows)}
    targets = sorted(env.targets, key=lambda t: order[t['group']])
    results = []
    details = []
    for index, target in enumerate(targets):
        emit('progress', current=index, total=len(targets), target=target['group'])
        obs, _ = env.reset(seed=810000 + index, options={'target': target})
        started = time.perf_counter()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        path = info['q_path']
        if not path or not np.isfinite(np.asarray(path)).all():
            raise ValueError(f'Invalid inference path: {target["group"]}')
        # Reward/SDF proximity is not a substitute for exact edge validation.
        bad_start = bool(env.backend.check_collision(env.robot, path[0]).collision)
        bad_edges = [i for i, (a, b) in enumerate(zip(path[:-1], path[1:]))
                     if env.backend.check_edge_collision(env.robot, a, b).collision]
        ok = info['result'] == 'success' and not bad_start and not bad_edges
        reason = '' if ok else (f'final_verification_failed: start={bad_start}, edges={bad_edges}'
                               if bad_start or bad_edges else f'SAC rollout: {info["result"]}')
        ref = by_group[target['group']]
        results.append({'group_name': target['group'], 'robot_name': env.robot, 'pose_name': ref['pose_name'],
                        'positioner_r_deg': float(ref.get('positioner_r_deg') or 0),
                        'q_path': path, 'status': 'success' if ok else 'failed', 'message': reason,
                        'collision_preview': not ok, 'elapsed': time.perf_counter() - started,
                        'verification': {'valid': not bad_start and not bad_edges, 'bad_edges': bad_edges}})
        details.append({'group': target['group'], 'rollout_result': info['result'], 'verified_success': ok,
                        'steps': info['steps'], 'shield_steps': info['shield_steps'], 'message': reason})
        emit('target', current=index + 1, total=len(targets), target=target['group'], success=ok, message=reason)
    out.mkdir(parents=True, exist_ok=False)
    summary = save_plan_run(out, results, meta={'planner': 'residual_sac', 'source': 'rl/infer_sac.py',
                            'model': str(Path(args.model).resolve()), 'shield': args.shield,
                            'reference_run': str(Path(args.reference).resolve()),
                            'snapshot': str(Path(args.snapshot).resolve()), 'skipped_targets': skipped},
                            joint_names_by_robot={env.robot: list(env.backend.joint_names(env.robot))})
    shutil.copy2(args.snapshot, out / 'scene_snapshot.pkl')
    (out / 'inference_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    (out / 'inference.json').write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding='utf-8')
    emit('complete', run_dir=str(out), n_success=summary['n_success'], n_failed=summary['n_failed'],
         n_skipped=len(skipped))


if __name__ == '__main__':
    main()
