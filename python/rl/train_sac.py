"""SAC training on rl/path_env.py (stable-baselines3), parallel environments, TensorBoard logs and checkpoints.

    python python/rl/train_sac.py --name dda_single --targets "Point 1 - Inspection pose 2" \
        --snapshots sample/planning_snapshot_p2_3.pkl --timesteps 150000
    tensorboard --logdir debug/rl

Logged: rollout/ep_rew_mean, rollout/success_rate (info["is_success"]), custom/collision_rate, custom/timeout_rate,
custom/ep_len, custom/reward_<term> (mean per episode), custom/exact_check_ratio. Output: debug/rl/<name>/
(tensorboard, checkpoints, final model, config used).
"""
import argparse
import collections
import json
import pathlib
import sys
import time
import statistics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rl.path_env import PathPlanningEnv, load_rl_config, ROOT_PATH  # noqa: E402


def make_env(cfg, snapshots, rank):
    def _init():
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        return PathPlanningEnv(cfg, snapshots=snapshots, seed=1000 + rank)
    return _init


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--robot", default=None)
    ap.add_argument("--snapshots", default=None, help="comma list (default: env.train_snapshots)")
    ap.add_argument("--targets", default=None, help="comma list of group names (default: all)")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--gradient-steps", type=int, default=None)
    ap.add_argument("--resume", default=None, help="Resume model/optimizer checkpoint; --timesteps is the cumulative target")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    cfg = load_rl_config(args.config)
    if args.robot:
        cfg["env"]["robot"] = args.robot
    if args.targets:
        cfg["env"]["target_filter"] = [t.strip() for t in args.targets.split(",") if t.strip()]
    sc = cfg["sac"]
    if args.timesteps:
        sc["total_timesteps"] = args.timesteps
    if args.n_envs:
        sc["n_envs"] = args.n_envs
    if args.gradient_steps:
        sc["gradient_steps"] = args.gradient_steps
    snapshots = [s.strip() for s in args.snapshots.split(",")] if args.snapshots else cfg["env"]["train_snapshots"]
    out = ROOT_PATH / "debug" / "rl" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"cfg": cfg, "snapshots": snapshots}, indent=2), encoding="utf-8")
    scene_labels = {pathlib.Path(p).name: f"s{i}" for i, p in enumerate(snapshots)}
    (out / "scene_log_names.json").write_text(json.dumps(scene_labels, indent=2), encoding="utf-8")

    torch.set_num_threads(2)
    t0 = time.perf_counter()
    venv = VecMonitor(SubprocVecEnv([make_env(cfg, snapshots, i) for i in range(int(sc["n_envs"]))]))
    print(f"{sc['n_envs']} envs up in {time.perf_counter() - t0:.0f}s", flush=True)

    class Stats(BaseCallback):
        """Episode outcome rates and mean reward terms, logged every rollout."""
        def __init__(self):
            super().__init__()
            self.buf = collections.deque(maxlen=200)
            self.t_last, self.n_last = time.perf_counter(), 0
            self.last_progress_step = 0
            from rl.convergence import Plateau
            self.stop_cfg = sc.get("early_stop", {})
            self.plateau = Plateau(self.stop_cfg.get("min_steps", 25000),
                                   self.stop_cfg.get("tolerance", 0.1), self.stop_cfg.get("patience", 3))
            self.window_episodes = []
            self.window_losses = []
            self.window_start = 0
            self.stop_reason = "step_limit"

        def _on_step(self):
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if done and "result" in info:
                    self.buf.append(info)
                    if self.stop_cfg:
                        self.window_episodes.append((sum(info["reward_terms"].values()), info["result"] == "success"))
            if self.stop_cfg:
                values = self.logger.name_to_value
                if "train/actor_loss" in values and "train/critic_loss" in values:
                    self.window_losses.append((float(values["train/actor_loss"]), float(values["train/critic_loss"])))
                if self.num_timesteps - self.window_start >= self.stop_cfg.get("window_steps", 5000):
                    if self.window_episodes and self.window_losses:
                        metrics = dict(reward=statistics.mean(x[0] for x in self.window_episodes),
                                       success=statistics.mean(x[1] for x in self.window_episodes),
                                       actor_loss=statistics.median(x[0] for x in self.window_losses),
                                       critic_loss=statistics.median(x[1] for x in self.window_losses))
                        stop, changes = self.plateau.update(self.num_timesteps, metrics)
                        with (out / "convergence.jsonl").open("a") as f:
                            f.write(json.dumps(dict(step=self.num_timesteps, **metrics, changes=changes,
                                                   stable_windows=self.plateau.streak, stop=stop)) + "\n")
                        if stop:
                            self.stop_reason = "reward_loss_plateau"
                            return False
                    self.window_episodes.clear()
                    self.window_losses.clear()
                    self.window_start = self.num_timesteps
            if self.num_timesteps - self.last_progress_step >= 1000:
                progress = {"timesteps": self.num_timesteps, "total": int(sc["total_timesteps"]),
                            "updates": int(getattr(self.model, "_n_updates", 0)), "updated": time.time()}
                tmp = out / "progress.tmp.json"
                tmp.write_text(json.dumps(progress), encoding="utf-8")
                tmp.replace(out / "progress.json")
                self.last_progress_step = self.num_timesteps
            return True

        def _on_rollout_end(self):
            if not self.buf:
                return
            n = len(self.buf)
            res = collections.Counter(i["result"] for i in self.buf)
            for k in ("success", "collision", "timeout"):
                self.logger.record(f"custom/{k}_rate", res.get(k, 0) / n)
            self.logger.record("custom/ep_len", sum(i["steps"] for i in self.buf) / n)
            self.logger.record("custom/augmented_episode_ratio", sum(i.get("n_boxes", 0) > 0 for i in self.buf) / n)
            self.logger.record("custom/obstacle_count", sum(i.get("n_boxes", 0) for i in self.buf) / n)
            for snapshot in sorted({i.get("snapshot", "unknown") for i in self.buf}):
                items = [i for i in self.buf if i.get("snapshot", "unknown") == snapshot]
                label = scene_labels.get(snapshot, "unknown")
                self.logger.record(f"scene/{label}_episodes", len(items))
                self.logger.record(f"scene/{label}_aug_ratio", sum(i.get("n_boxes", 0) > 0 for i in items) / len(items))
            self.logger.record("custom/curriculum_level", sum(i.get("curriculum_level", 1.0) for i in self.buf) / n)
            for k in self.buf[-1]["reward_terms"]:
                self.logger.record(f"custom/reward_{k}", sum(i["reward_terms"].get(k, 0.0) for i in self.buf) / n)
            ex = sum(i.get("exact_checks", 0) for i in self.buf)
            fa = sum(i.get("fast_checks", 0) for i in self.buf)
            self.logger.record("custom/exact_check_ratio", ex / max(1, ex + fa))
            now = time.perf_counter()
            self.logger.record("custom/env_steps_per_s", (self.num_timesteps - self.n_last) / max(1e-6, now - self.t_last))
            self.t_last, self.n_last = now, self.num_timesteps

    if args.resume:
        model = SAC.load(args.resume, env=venv, device=sc["device"], tensorboard_log=str(out / "tb"))
        # Legacy checkpoints contain weights/optimizers but no replay buffer.
        # Refill it before updating, using a threshold in cumulative environment steps.
        model.learning_starts = model.num_timesteps + int(sc["learning_starts"])
        print(f"RESUME checkpoint={args.resume} steps={model.num_timesteps} updates={model._n_updates}; "
              f"replay buffer unavailable, refill until step {model.learning_starts}", flush=True)
    else:
        model = SAC("MlpPolicy", venv, policy_kwargs={"net_arch": sc["net_arch"]}, learning_rate=sc["learning_rate"],
                buffer_size=sc["buffer_size"], batch_size=sc["batch_size"], gamma=sc["gamma"], tau=sc["tau"],
                ent_coef=sc["ent_coef"], train_freq=sc["train_freq"], gradient_steps=sc["gradient_steps"],
                learning_starts=sc["learning_starts"], device=sc["device"], verbose=1, seed=0,
                tensorboard_log=str(out / "tb"))
    stats = Stats()
    cb = CallbackList([stats, CheckpointCallback(save_freq=max(1, int(sc.get("checkpoint_steps", 50000)) // int(sc["n_envs"])),
                                                   save_path=str(out / "checkpoints"), name_prefix="sac")])
    try:
        remaining = max(0, int(sc["total_timesteps"]) - model.num_timesteps)
        if remaining:
            model.learn(remaining, callback=cb, log_interval=20, progress_bar=False,
                        reset_num_timesteps=not bool(args.resume))
        model.save(str(out / "final_model"))
        (out / "training_result.json").write_text(json.dumps({"timesteps": model.num_timesteps,
            "stop_reason": stats.stop_reason, "elapsed_seconds": time.perf_counter() - t0}, indent=2))
        print(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {out}", flush=True)
    finally:
        venv.close()


if __name__ == "__main__":
    main()
