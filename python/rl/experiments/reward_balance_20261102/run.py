"""Three fresh SAC runs with bounded APF, plateau stopping and matched held-out evaluation."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'debug/rl/reward_balance_20261102'
BASE = ROOT / 'python/rl/experiments/apf10_d030_aug_20261101'
CANDIDATES = [('a_collision7500', 10, 5, 7500), ('b_collision10000', 10, 5, 10000),
              ('c_apf5_collision7500', 5, 10, 7500)]


def summary():
    lines = ['# Reward balance pilot', '', 'All rates include failed targets. Reward values across different reward functions are not directly ranked.',
             'Path-safe retains the existing mandatory geometry checks, including side constraints; only the side reward was disabled.', '',
             '| candidate | steps / stop | evaluation | reached | clearance >= 0 and reached | path-safe | targets |',
             '|---|---|---|---:|---:|---:|---:|']
    for name, *_ in CANDIDATES:
        d = OUT / name
        result = json.loads((d/'training_result.json').read_text()) if (d/'training_result.json').exists() else {}
        for f in sorted(d.glob('eval_*_[0-9]*.json')):
            if '.partial.' in f.name:
                continue
            rows = json.loads(f.read_text())
            reached = [r for r in rows if r['result'] == 'success']
            clear = [r for r in reached if r.get('C_eff') is not None and r['C_eff'] >= 0]
            safe = [r for r in clear if r.get('mandatory_ok')]
            lines.append(f"| {name} | {result.get('timesteps', '?')} / {result.get('stop_reason', '?')} | {f.stem} | {len(reached)} | {len(clear)} | {len(safe)} | {len(rows)} |")
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, OUT/'run.py')
    sources = OUT/'sources'
    sources.mkdir()
    for name in ('path_env.py', 'train_sac.py', 'eval_sac.py', 'convergence.py'):
        shutil.copy2(ROOT/'python/rl'/name, sources/name)
    def run(label, args, log):
        (OUT/'status.json').write_text(json.dumps(dict(state='running', stage=label, updated=time.time())))
        print('START', label, flush=True)
        with log.open('w') as f:
            subprocess.run([sys.executable, '-u', *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        print('DONE', label, flush=True)
    try:
        for name, weight, clip, collision in CANDIDATES:
            d = OUT/name
            d.mkdir()
            cfg = json.loads((BASE/'train.json').read_text())
            cfg['reward'].update(side=0.0, apf=weight, apf_clip=clip, collision=collision)
            cfg['sac'].update(total_timesteps=60000, early_stop=dict(min_steps=25000, window_steps=5000, tolerance=0.1, patience=3))
            (d/'train.json').write_text(json.dumps(cfg, indent=2))
            run(name+'/train', ['python/rl/train_sac.py', '--name', 'reward_balance_20261102/'+name,
                               '--config', str(d/'train.json')], d/'train.log')
            for kind, seeds in [('clean', [810000]), ('aug', [910000,920000,930000])]:
                for method in ('sac','shield'):
                    ecfg = json.loads((BASE/f'eval_{kind}{"_shield" if method == "shield" else ""}.json').read_text())
                    ecfg['reward'] = copy.deepcopy(cfg['reward'])
                    ecfg['sac']['device'] = 'cpu'
                    ep = d/f'config_eval_{kind}_{method}.json'
                    ep.write_text(json.dumps(ecfg, indent=2))
                    for seed in seeds:
                        label = f'eval_{kind}_{method}_{seed}'
                        run(name+'/'+label, ['python/rl/eval_sac.py', '--model', str(d/'final_model.zip'),
                            '--config', str(ep), '--snapshots', ','.join(cfg['env']['eval_snapshots']),
                            '--seed', str(seed), '--stochastic', '0', '--compare', '', '--out', str(d/f'{label}.json')], d/f'{label}.log')
                        summary()
        (OUT/'status.json').write_text(json.dumps(dict(state='complete', updated=time.time())))
    except BaseException as e:
        (OUT/'status.json').write_text(json.dumps(dict(state='failed', error=str(e), updated=time.time())))
        raise

if __name__ == '__main__':
    main()
