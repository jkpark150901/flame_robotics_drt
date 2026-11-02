"""Evaluate a saved run on held-out clean/augmented scenes with matched seeds."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'python/rl/experiments/apf10_d030_aug_20261101'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--model',type=Path,help='Default: RUN/final_model.zip')
    p.add_argument('--out',type=Path,help='New output directory; default RUN/evaluation')
    p.add_argument('--methods',nargs='+',choices=['sac','shield','zero'],default=['sac','shield','zero'])
    p.add_argument('--aug-seeds',type=int,nargs='+',default=[910000,920000,930000])
    p.add_argument('--clean-seed',type=int,default=810000)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();run=a.run.resolve();out=(a.out or run/'evaluation').resolve()
    train=json.loads((run/'train.json').read_text());train=train.get('cfg',train)
    model=(a.model or run/'final_model.zip').resolve()
    if not a.prepare_only and any(m!='zero' for m in a.methods) and not model.is_file():p.error(f'missing model: {model}')
    out.mkdir(parents=True,exist_ok=False)
    lines=['# Held-out evaluation','','Clearance uses reached paths only. Path-safe = reached + C_eff >= 0 + mandatory_ok.','','| condition | reached | path-safe | mean/min C_eff (mm) |','|---|---:|---:|---:|']
    commands=[]
    for kind,seeds in [('clean',[a.clean_seed]),('aug',a.aug_seeds)]:
        for method in dict.fromkeys(a.methods):
            cfg=copy.deepcopy(train)
            # Preserve this run\'s split/reference sources; evaluation has no start/reference augmentation.
            env=cfg['env'];env['target_split']['split']='eval'
            env['start_perturb_prob']=0.;env['curriculum']={'enabled':False}
            env['residual'].update(random_reference=False,mid_start_prob=0.,start_noise_prob=0.,shield=method=='shield')
            # Canonical fixed RRTConnect references shared with the historical held-out evaluation.
            env['residual']['reference_runs']=json.loads((BASE/'eval_clean.json').read_text())['env']['residual']['reference_runs']
            env.setdefault('random_obstacles',{}).update(enabled=kind=='aug',prob=1.)
            cfg['sac']['device']=a.device
            config=out/f'config_{kind}_{method}.json';config.write_text(json.dumps(cfg,indent=2))
            for seed in dict.fromkeys(seeds):
                label=f'{kind}_{method}_{seed}';dest=out/f'{label}.json'
                cmd=[sys.executable,'-u',str(ROOT/'python/rl/eval_sac.py'),'--model','zero' if method=='zero' else str(model),'--config',str(config),'--snapshots',','.join(env['eval_snapshots']),'--stochastic','0','--compare','','--seed',str(seed),'--device',a.device,'--out',str(dest)]
                commands.append(cmd);(out/'commands.json').write_text(json.dumps(commands,indent=2))
                if a.prepare_only:continue
                print('Evaluating',label,flush=True)
                with (out/f'{label}.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                rows=json.loads(dest.read_text());ok=[r for r in rows if r['result']=='success']
                safe=[r for r in ok if r.get('C_eff') is not None and r['C_eff']>=0 and r.get('mandatory_ok')]
                distances=[r['C_eff']*1000 for r in ok if r.get('C_eff') is not None]
                clearance=f'{statistics.mean(distances):.1f}/{min(distances):.1f}' if distances else '-'
                lines.append(f'| {label} | {len(ok)}/{len(rows)} | {len(safe)}/{len(rows)} | {clearance} |')
                (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Output:',out)


if __name__=='__main__':main()
