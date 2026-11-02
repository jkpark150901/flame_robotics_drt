"""Stage-1 checks of rl/path_env.py: SB3 env checker, step speed, and a straight-line controller (full speed toward
the goal) on every target - success / collision / timeout per target and the reward terms, to see that the reward
signs and sizes match the path metrics before training.

    python python/rl/check_env.py [--robot dda_rb10_1300e] [--split train]
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rl.path_env import PathPlanningEnv, load_rl_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--random-steps", type=int, default=300)
    ap.add_argument("--sac-smoke", type=int, default=0, help="also run a short SAC training of this many steps")
    args = ap.parse_args()
    cfg = load_rl_config()
    if args.robot:
        cfg["env"]["robot"] = args.robot
    t0 = time.perf_counter()
    env = PathPlanningEnv(cfg, split=args.split, seed=0)
    print(f"env built in {time.perf_counter() - t0:.1f}s: robot={env.robot}, targets={len(env.targets)}, "
          f"obs={env.observation_space.shape}, act={env.action_space.shape}, max_step={np.round(env.max_step, 3).tolist()}")

    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    print("SB3 check_env: ok")

    # speed: random policy
    obs, _ = env.reset(seed=1)
    t0, n, eps = time.perf_counter(), 0, 0
    while n < args.random_steps:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        n += 1
        if term or trunc:
            eps += 1
            env.reset()
    dt = time.perf_counter() - t0
    print(f"random policy: {n} steps in {dt:.1f}s = {n / dt:.0f} steps/s per env, {eps} episodes ended")

    # straight-line controller on every target from its planning start posture
    rows = []
    env.cfg["env"]["start_perturb_prob"] = 0.0   # planning start posture (retreat posture if the stored one collides)
    for target in env.targets:
        obs, _ = env.reset(options={"target": target})
        ret, t0 = 0.0, time.perf_counter()
        while True:
            a = np.clip((target["goal_q"] - env.q) / env.max_step, -1.0, 1.0)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            if term or trunc:
                break
        rows.append((target, info, ret, time.perf_counter() - t0))
    print("\nstraight-line controller (full speed toward the goal):")
    for target, info, ret, el in rows:
        terms = info["reward_terms"]
        print(f"  {pathlib.Path(target['snapshot']).stem[-4:]} {target['group']:28s} ph{target['phase']} "
              f"{info['result']:9s} steps {info['steps']:3d} return {ret:8.1f} | "
              + " ".join(f"{k} {v:7.1f}" for k, v in terms.items() if abs(v) > 1e-9) + f" | {el:.1f}s")
    res = [r[1]["result"] for r in rows]
    print(f"\nsummary: success {res.count('success')}/{len(res)}, collision {res.count('collision')}, "
          f"timeout {res.count('timeout')}")

    if args.sac_smoke:
        from stable_baselines3 import SAC
        sc = cfg["sac"]
        env.cfg["env"]["start_perturb_prob"] = float(cfg["env"]["start_perturb_prob"])
        model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": sc["net_arch"]}, learning_rate=sc["learning_rate"],
                    batch_size=256, learning_starts=500, device=sc["device"], verbose=0, seed=0)
        t0 = time.perf_counter()
        model.learn(args.sac_smoke)
        print(f"SAC smoke: {args.sac_smoke} steps on {model.device} in {time.perf_counter() - t0:.0f}s "
              f"({args.sac_smoke / (time.perf_counter() - t0):.0f} steps/s incl. gradient updates)")


if __name__ == "__main__":
    main()
