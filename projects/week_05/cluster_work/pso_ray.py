import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import ray

# Підключення зовні кластера (порт-проксі або NodePort):
RAY_ADDR = os.environ.get("RAY_ADDRESS", "ray://127.0.0.1:30001")

@ray.remote
class Particle:
    def __init__(self, dim: int, bounds: Tuple[float, float], w: float, c1: float, c2: float, seed=None):
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.lb, self.ub = bounds
        self.w, self.c1, self.c2 = w, c1, c2
        self.pos = self.rng.uniform(self.lb, self.ub, size=dim)
        self.vel = self.rng.uniform(-abs(self.ub - self.lb), abs(self.ub - self.lb), size=dim) * 0.1
        self.best_pos = self.pos.copy()
        self.best_val = self._rastrigin(self.pos)

    def _rastrigin(self, x: np.ndarray) -> float:
        # Мінімізуємо f(x) = 10d + sum(x_i^2 - 10 cos(2π x_i))
        d = x.shape[0]
        return 10 * d + np.sum(x**2 - 10 * np.cos(2 * math.pi * x))

    def evaluate(self):
        val = self._rastrigin(self.pos)
        if val < self.best_val:
            self.best_val = val
            self.best_pos = self.pos.copy()
        return val, self.pos.copy(), self.best_val, self.best_pos.copy()

    def step(self, gbest_pos: np.ndarray):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        cognitive = self.c1 * r1 * (self.best_pos - self.pos)
        social = self.c2 * r2 * (gbest_pos - self.pos)
        self.vel = self.w * self.vel + cognitive + social
        self.pos = np.clip(self.pos + self.vel, self.lb, self.ub)

    def get_state(self):
        return self.pos.copy(), self.vel.copy(), self.best_pos.copy(), self.best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lb", type=float, default=-5.12)
    parser.add_argument("--ub", type=float, default=5.12)
    parser.add_argument("--w", type=float, default=0.72)
    parser.add_argument("--c1", type=float, default=1.49)
    parser.add_argument("--c2", type=float, default=1.49)
    args = parser.parse_args()

    ray.init(address=RAY_ADDR, ignore_reinit_error=True)

    bounds = (args.lb, args.ub)
    particles = [
        Particle.remote(args.dim, bounds, args.w, args.c1, args.c2, seed=i)
        for i in range(args.particles)
    ]

    # Ініціалізація
    evals = ray.get([p.evaluate.remote() for p in particles])
    gbest_val = float("inf")
    gbest_pos = None
    for val, pos, pbest_val, pbest_pos in evals:
        if pbest_val < gbest_val:
            gbest_val, gbest_pos = pbest_val, pbest_pos

    start = time.time()
    for t in range(args.iters):
        # Крок оновлення (паралельно)
        ray.get([p.step.remote(gbest_pos) for p in particles])
        evals = ray.get([p.evaluate.remote() for p in particles])
        improved = False
        for val, pos, pbest_val, pbest_pos in evals:
            if pbest_val < gbest_val:
                gbest_val, gbest_pos = pbest_val, pbest_pos
                improved = True
        if (t + 1) % 10 == 0 or improved:
            print(f"iter={t+1:04d}  gbest={gbest_val:.6f}")

    dur = time.time() - start
    print("\nDone:")
    print("  Best value:", gbest_val)
    print("  Best position:", np.array2string(gbest_pos, precision=4))
    print(f"  Time: {dur:.2f}s for {args.iters} iterations with {args.particles} particles")


if __name__ == "__main__":
    main()