#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI demo: Monte Carlo VaR для простого портфеля з 3 активів.
Показує Broadcast параметрів, паралельну симуляцію, та дві стратегії агрегації:
- mode=gather: збираємо всі локальні P&L на rank 0 і рахуємо квантиль точно
- mode=hist:   об'єднуємо гістограми через Allreduce (масштабовано)
"""

from mpi4py import MPI
import numpy as np
import argparse
import math
import time


def simulate_pnl(n_paths, S0, mu, cov, dt, weights, rng):
    """
    Симуляція 1-денного P&L портфеля з 3 активів:
      log-returns ~ N(mu*dt, cov*dt), ціни Lognormal.
    P&L = w · (S1 - S0)
    """
    d = len(S0)
    L = np.linalg.cholesky(cov * dt)        # Cholesky для корельованих нормалок
    Z = rng.standard_normal(size=(n_paths, d))
    R = Z @ L.T + (mu * dt)                 # корельовані нормальні дохідності
    S1 = S0 * np.exp(R)                     # один крок GBM
    pnl = (S1 - S0) @ weights               # вектор P&L по шляхах
    return pnl


def quantile_from_hist(edges, counts, q):
    """
    Обчислення квантіля q з глобальної гістограми (edges, counts).
    Повертає значення порогу P&L для даного квантіля.
    """
    cdf = np.cumsum(counts)
    total = cdf[-1]
    if total == 0:
        return np.nan
    target = q * total
    idx = np.searchsorted(cdf, target, side="left")
    idx = min(max(idx, 0), len(edges) - 2)
    # лінійна інтерполяція всередині біну
    left_count_before = cdf[idx - 1] if idx > 0 else 0
    in_bin = counts[idx]
    if in_bin <= 0:
        return edges[idx]
    frac = (target - left_count_before) / in_bin
    return edges[idx] + frac * (edges[idx + 1] - edges[idx])


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="MPI Monte Carlo VaR demo")
    parser.add_argument("--paths", type=int, default=300_000, help="Кількість симульованих шляхів (загалом)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Ліва хвостова ймовірність (наприклад 0.05 для VaR 95%)")
    parser.add_argument("--mode", choices=["gather", "hist"], default="hist",
                        help="Спосіб агрегації: gather (точно) або hist (масштабовано)")
    parser.add_argument("--bins", type=int, default=400, help="Кількість бінів для гістограми в режимі hist")
    parser.add_argument("--seed", type=int, default=42, help="Базове зерно RNG")
    args = parser.parse_args()

    # ------------------------------
    # 1) Вихідні параметри (задає rank 0) і розсилка всім
    # ------------------------------
    if rank == 0:
        # 3 активи, умовно: акції/крипто/облігації
        S0 = np.array([100.0, 30_000.0, 1000.0])   # ціни сьогодні
        weights = np.array([50.0, 0.003, 2.0])     # позиції (штук/лотів); для BTC вага мала
        mu = np.array([0.05, 0.10, 0.02])          # очікувана річна дохідність
        vol = np.array([0.20, 0.60, 0.05])         # річна волатильність
        rho = np.array([
            [ 1.0,  0.2, -0.1],
            [ 0.2,  1.0,  0.0],
            [-0.1,  0.0,  1.0],
        ])
        cov = np.outer(vol, vol) * rho             # коваріаційна матриця
        dt = 1.0 / 252.0                           # 1 торговий день
        params = dict(S0=S0, mu=mu, cov=cov, dt=dt, weights=weights)
    else:
        params = None

    params = comm.bcast(params, root=0)
    S0, mu, cov, dt, weights = params["S0"], params["mu"], params["cov"], params["dt"], params["weights"]

    # ------------------------------
    # 2) Розподіл кількості шляхів
    # ------------------------------
    total_paths = args.paths
    base = total_paths // size
    extra = total_paths % size
    n_local = base + (1 if rank < extra else 0)

    # ------------------------------
    # 3) Паралельна симуляція
    # ------------------------------
    rng = np.random.default_rng(args.seed + rank)
    t0 = time.time()
    pnl_local = simulate_pnl(n_local, S0, mu, cov, dt, weights, rng)
    t1 = time.time()

    # Корисні глобальні статистики через Allreduce (середнє, σ, min/max)
    local_sum = np.array([pnl_local.sum()], dtype="float64")
    local_sumsq = np.array([np.square(pnl_local).sum()], dtype="float64")
    local_cnt = np.array([pnl_local.size], dtype="int64")
    global_sum = np.array([0.0], dtype="float64")
    global_sumsq = np.array([0.0], dtype="float64")
    global_cnt = np.array([0], dtype="int64")

    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_sumsq, global_sumsq, op=MPI.SUM)
    comm.Allreduce(local_cnt, global_cnt, op=MPI.SUM)

    mean_pnl = global_sum[0] / max(1, global_cnt[0])
    var_pnl = global_sumsq[0] / max(1, global_cnt[0]) - mean_pnl**2
    std_pnl = math.sqrt(max(var_pnl, 0.0))

    # ------------------------------
    # 4) Обчислення VaR
    # ------------------------------
    alpha = args.alpha
    mode = args.mode

    if mode == "gather":
        # Збираємо всі локальні P&L на root
        all_pnls = comm.gather(pnl_local, root=0)
        if rank == 0:
            pnl = np.concatenate(all_pnls) if all_pnls else np.empty((0,))
            q = np.quantile(pnl, alpha)
            var = -q  # VaR – це позитивне число (втрати)
            t2 = time.time()
            print(f"[rank 0] MPI size={size}, total_paths={total_paths}, mode={mode}")
            print(f"Sim time: {t1 - t0:.3f}s, Quantile time: {t2 - t1:.3f}s")
            print(f"Mean P&L={mean_pnl:.4f}, Std={std_pnl:.4f}")
            print(f"VaR({100*(1-alpha):.0f}%) = {-q:.4f} (left {alpha*100:.1f}% quantile = {q:.4f})")

    else:  # mode == "hist"
        # Глобальні межі для гістограми
        local_min = np.array([pnl_local.min() if pnl_local.size else np.inf], dtype="float64")
        local_max = np.array([pnl_local.max() if pnl_local.size else -np.inf], dtype="float64")
        global_min = np.array([0.0], dtype="float64")
        global_max = np.array([0.0], dtype="float64")
        comm.Allreduce(local_min, global_min, op=MPI.MIN)
        comm.Allreduce(local_max, global_max, op=MPI.MAX)

        # Невеликий запас по краях
        span = max(global_max[0] - global_min[0], 1e-9)
        lo = global_min[0] - 0.02 * span
        hi = global_max[0] + 0.02 * span
        edges = np.linspace(lo, hi, args.bins + 1)

        # Локальна гістограма
        counts_local, _ = np.histogram(pnl_local, bins=edges)
        counts_local = counts_local.astype(np.int64)

        # Глобальна через Allreduce
        counts_global = np.zeros_like(counts_local)
        comm.Allreduce(counts_local, counts_global, op=MPI.SUM)

        if rank == 0:
            q_val = quantile_from_hist(edges, counts_global, alpha)
            var = -q_val
            t2 = time.time()
            print(f"[rank 0] MPI size={size}, total_paths={total_paths}, mode={mode}, bins={args.bins}")
            print(f"Sim time: {t1 - t0:.3f}s, Hist+quantile time: {t2 - t1:.3f}s")
            print(f"Mean P&L={mean_pnl:.4f}, Std={std_pnl:.4f}")
            print(f"VaR({100*(1-alpha):.0f}%) ≈ {var:.4f} (approx; q≈{q_val:.4f})")


if __name__ == "__main__":
    main()
