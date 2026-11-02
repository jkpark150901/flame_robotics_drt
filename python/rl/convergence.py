"""Windowed plateau heuristic; stability is not proof of SAC convergence."""
import math
import statistics


class Plateau:
    def __init__(self, minimum=25000, tolerance=0.1, patience=3):
        self.minimum, self.tolerance, self.patience = minimum, tolerance, patience
        self.previous = None
        self.streak = 0

    def update(self, step, metrics):
        valid = all(math.isfinite(v) for v in metrics.values())
        changes = {}
        if valid and self.previous is not None:
            for key in ('reward', 'actor_loss', 'critic_loss'):
                a, b = metrics[key], self.previous[key]
                changes[key] = abs(a-b) / max(abs(a), abs(b), 1.0)
            stable = max(changes.values()) <= self.tolerance and abs(metrics['success']-self.previous['success']) <= 0.05
            self.streak = self.streak + 1 if stable and step >= self.minimum else 0
        else:
            self.streak = 0
        self.previous = metrics if valid else None
        return self.streak >= self.patience, changes
