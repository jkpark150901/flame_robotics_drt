"""Create an isolated reward configuration and run fresh SAC training."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'python/rl/experiments/apf10_d030_aug_20261101/train.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True, help='New directory name under debug/rl')
    p.add_argument('--config', type=Path, default=BASE)
    p.add_argument('--reward', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--apf-max-penalty', type=float, default=50)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--min-steps', type=int, default=25000)
    p.add_argument('--window-steps', type=int, default=5000)
    p.add_argument('--tolerance', type=float, default=.1)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--no-early-stop', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--keep-config-reward', action='store_true',
                   help='Use reward defaults from --config before applying --reward overrides')
    a = p.parse_args()
    if Path(a.name).name != a.name or a.name in ('.', '..'):
        p.error('--name must be a single directory name')
    if min(a.steps,a.min_steps,a.window_steps,a.patience) <= 0 or not 0 <= a.tolerance <= 1:
        p.error('step counts/patience must be positive; tolerance must be in [0,1]')
    if not math.isfinite(a.apf_max_penalty) or a.apf_max_penalty < 0:
        p.error('APF maximum must be finite and nonnegative')
    cfg = json.loads(a.config.resolve().read_text());cfg = cfg.get('cfg',cfg)
    if not a.keep_config_reward:
        cfg['reward'].update(side=0.,goal=1000.,collision=7500.,apf=10.)
    for item in a.reward:
        try:
            key,value = item.split('=',1);value=float(value)
        except ValueError:
            p.error(f'invalid reward: {item}')
        if key not in cfg['reward'] or isinstance(cfg['reward'][key],dict) or key=='apf_clip':
            p.error(f'unsupported reward key {key}; use --apf-max-penalty for clipping')
        if not math.isfinite(value) or value < 0 or (key=='apf_d0' and value==0):
            p.error(f'invalid reward value: {item}')
        cfg['reward'][key]=value
    cfg['reward']['apf_clip'] = a.apf_max_penalty/cfg['reward']['apf'] if cfg['reward']['apf'] else 0.
    cfg['sac'].update(total_timesteps=a.steps,device=a.device,checkpoint_steps=5000)
    cfg['sac'].pop('early_stop',None)
    if not a.no_early_stop:
        cfg['sac']['early_stop']=dict(min_steps=a.min_steps,window_steps=a.window_steps,tolerance=a.tolerance,patience=a.patience)
    out=ROOT/'debug/rl'/a.name
    out.mkdir(parents=True,exist_ok=False)
    path=out/'train.json';path.write_text(json.dumps(cfg,indent=2))
    command=[sys.executable,'-u',str(ROOT/'python/rl/train_sac.py'),'--name',a.name,'--config',str(path)]
    (out/'command.json').write_text(json.dumps(command,indent=2))
    print(f'Configuration: {path}',flush=True)
    if not a.prepare_only:
        subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
