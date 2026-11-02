"""Run training followed by clean and independently seeded augmented held-out evaluations.

Run with the flame interpreter from any directory. GPU selection is inherited from
CUDA_VISIBLE_DEVICES. Existing output directories are refused to protect experiments.
"""
import collections
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'debug/rl/dda_apf10_d030_split_aug_20261101'


def summarize():
    lines = ['# APF 10 / 0.30 m, per-snapshot held-out evaluation', '',
             'Rates use all requested targets; distance and timing statistics use successful rollouts only.',
             'Augmented runs use new scene seeds, shared by SAC, SAC+shield and zero-policy baseline.',
             'Rollout time excludes reference planning and training. Zero policy follows the saved reference.', '',
             '| evaluation | scene | targets | reached | path safe | augmented | clearance mean/min mm | motion s | rollout s |',
             '|---|---|---:|---:|---:|---:|---|---:|---:|']
    for path in sorted(OUT.glob('eval_*_[0-9]*.json')):
        if path.name.endswith('.partial.json'):
            continue
        rows = json.loads(path.read_text())
        groups = {'ALL': rows}
        for row in rows:
            groups.setdefault(row['snapshot'], []).append(row)
        for scene, items in groups.items():
            ok = [r for r in items if r['result'] == 'success']
            safe = sum(r.get('mandatory_ok') and r.get('C_eff') is not None and r['C_eff'] >= 0 for r in ok)
            ce = [r['C_eff'] * 1000 for r in ok if r.get('C_eff') is not None]
            clearance = f'{statistics.mean(ce):.1f}/{min(ce):.1f}' if ce else '-'
            def mean(key):
                values = [r[key] for r in ok if r.get(key) is not None]
                return f'{statistics.mean(values):.2f}' if values else '-'
            lines.append(f'| {path.stem} | {scene} | {len(items)} | {len(ok)} | {safe} | '
                         f'{sum(r.get("n_boxes", 0) > 0 for r in items)} | {clearance} | '
                         f'{mean("T_motion")} | {mean("t_plan")} |')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', help='Model checkpoint to resume in the existing experiment directory')
    parser.add_argument('--eval-only', action='store_true', help='Continue evaluation using the final model')
    args = parser.parse_args()
    continuation = bool(args.resume or args.eval_only)
    if args.eval_only:
        assert (OUT / 'final_model.zip').is_file(), 'No final model for evaluation'
    if args.resume:
        args.resume = str(Path(args.resume).resolve(strict=True))
    if continuation:
        assert OUT.is_dir(), 'Resume requires existing experiment output'
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        for p in HERE.glob('*.json'):
            shutil.copy2(p, OUT / p.name)
    cfg = json.loads((OUT / 'train.json').read_text())
    inputs = [ROOT / p for p in cfg['env']['train_snapshots']]
    sources = [ROOT / 'python/rl' / f for f in ('path_env.py', 'train_sac.py', 'eval_sac.py')]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    source_dir = OUT / (f'sources_resume_{stamp}' if continuation else 'sources')
    source_dir.mkdir()
    for p in sources:
        shutil.copy2(p, source_dir / p.name)
    manifest = {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'resume': args.resume, 'eval_only': args.eval_only,
                'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in inputs + sources}, 'commands': []}
    def run(label, command_args):
        command = [sys.executable, '-u', *command_args]
        manifest['commands'].append({'label': label, 'argv': command})
        (OUT / (f'manifest_resume_{stamp}.json' if continuation else 'manifest.json')).write_text(json.dumps(manifest, indent=2))
        (OUT / 'status.json').write_text(json.dumps({'stage': label, 'state': 'running', 'updated': time.time()}))
        print(f'START {label}', flush=True)
        with (OUT / f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'DONE {label}', flush=True)
    try:
        train_args = ['python/rl/train_sac.py', '--name', OUT.name, '--config', str(OUT / 'train.json')]
        if args.resume:
            train_args += ['--resume', args.resume]
        if not args.eval_only:
            run(f'train_resume_{stamp}' if args.resume else 'train', train_args)
        summarize()
        snapshots = ','.join(cfg['env']['eval_snapshots'])
        # The same seed reproduces the same boxes independently of the policy and shield.
        for augmented, seeds in [(False, [810000]), (True, [910000, 920000, 930000])]:
            kind = 'aug' if augmented else 'clean'
            for seed in seeds:
                for method in ['sac', 'shield', 'zero']:
                    config = OUT / f'eval_{kind}{"_shield" if method == "shield" else ""}.json'
                    label = f'eval_{kind}_{method}_{seed}'
                    if args.eval_only and (OUT / f'{label}.json').is_file():
                        print(f'SKIP completed {label}', flush=True)
                        continue
                    run(label, ['python/rl/eval_sac.py', '--model',
                                'zero' if method == 'zero' else str(OUT / 'final_model.zip'),
                                '--config', str(config), '--snapshots', snapshots, '--seed', str(seed),
                                '--stochastic', '0' if augmented or method == 'zero' else '3',
                                '--compare', '' if augmented else 'RRTConnect_none,RRTConnect_chomp,RRTConnect_stomp',
                                '--out', str(OUT / f'{label}.json')])
                    summarize()
        (OUT / 'status.json').write_text(json.dumps({'state': 'complete', 'updated': time.time()}))
        print('ALL_DONE', flush=True)
    except BaseException as exc:
        (OUT / 'status.json').write_text(json.dumps({'state': 'failed', 'error': str(exc), 'updated': time.time()}))
        raise


if __name__ == '__main__':
    main()
