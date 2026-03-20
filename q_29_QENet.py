"""
QENet - Quantum Elastic Net
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/Users/4c/Desktop/GHQ/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
ALPHA = 0.005
L1_RATIO = 0.5
MAX_ITER = 300


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_quantum_kernel():
    n_states = 1 << NUM_QUBITS
    fmap = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)

    statevectors = []
    for v in range(n_states):
        feat = value_to_features(v)
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        statevectors.append(sv)

    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i, n_states):
            fid = abs(statevectors[i].inner(statevectors[j])) ** 2
            K[i, j] = fid
            K[j, i] = fid

    return K


def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def quantum_elastic_net(K, y, alpha=ALPHA, l1_ratio=L1_RATIO,
                        max_iter=MAX_ITER):
    n = K.shape[0]
    w = np.zeros(n)
    l1_pen = alpha * l1_ratio
    l2_pen = alpha * (1.0 - l1_ratio)

    for iteration in range(max_iter):
        for j in range(n):
            r_j = y - K @ w + K[:, j] * w[j]
            z_j = np.dot(K[:, j], r_j)
            denom = np.dot(K[:, j], K[:, j]) + l2_pen + 1e-10
            w[j] = soft_threshold(z_j / denom, l1_pen / denom)

    pred = K @ w
    return pred, w


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Kvantni kernel (ZZFeatureMap, {NUM_QUBITS}q, reps=1) ---")
    K = compute_quantum_kernel()
    print(f"  Kernel matrica: {K.shape}, rang: {np.linalg.matrix_rank(K)}")

    print(f"\n--- QElasticNet po pozicijama (alpha={ALPHA}, "
          f"l1_ratio={L1_RATIO}, {MAX_ITER} iter) ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred, w = quantum_elastic_net(K, y)

        n_nonzero = np.sum(np.abs(w) > 1e-8)
        l1_norm = np.sum(np.abs(w))
        l2_norm = np.sqrt(np.sum(w ** 2))
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1}: nz={n_nonzero} "
              f"L1={l1_norm:.3f} L2={l2_norm:.3f}  {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QENet, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Kvantni kernel (ZZFeatureMap, 5q, reps=1) ---
  Kernel matrica: (32, 32), rang: 32

--- QElasticNet po pozicijama (alpha=0.005, l1_ratio=0.5, 300 iter) ---
  Poz 1: nz=27 L1=0.857 L2=0.250  1:0.151 | 2:0.132 | 3:0.116
  Poz 2: nz=28 L1=0.711 L2=0.164  8:0.088 | 5:0.078 | 9:0.077
  Poz 3: nz=30 L1=0.640 L2=0.139  13:0.066 | 12:0.064 | 14:0.063
  Poz 4: nz=26 L1=0.546 L2=0.128  21:0.067 | 23:0.066 | 18:0.066
  Poz 5: nz=29 L1=0.620 L2=0.146  29:0.066 | 26:0.065 | 27:0.064
  Poz 6: nz=30 L1=0.804 L2=0.188  33:0.085 | 32:0.083 | 35:0.082
  Poz 7: nz=28 L1=1.037 L2=0.293  7:0.185 | 38:0.155 | 37:0.134

==================================================
Predikcija (QENet, deterministicki, seed=39):
[1, 8, x, y, z, 33, 38]
==================================================
"""



"""
QENet - Quantum Elastic Net

QENet je kvantni algoritam za regresiju sa L1 i L2 regularizacijom.
QENet se sastoji od 5 qubita i 1 sloja Ry+CX+Rz rotacija.

Isti kvantni kernel (ZZFeatureMap, fidelity, 5 qubita)
Elastic Net = kombinacija L1 (Lasso) + L2 (Ridge) penalizacije
l1_ratio=0.5 - balans izmedju sparsity-ja (L1) i stabilnosti (L2)
Coordinate descent sa 300 iteracija
Pokazuje L1 normu, L2 normu i broj nenula - dijagnostika modela
Prednost nad cistim Lasso: stabilniji kad su kernel kolone korelisane
Prednost nad cistim Ridge: zadrzava sparsity (bira bitne vrednosti)
Deterministicki, brz.
"""
