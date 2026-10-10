"""What would the first measured batch teach?  A preposterior (synthetic) experiment.  MODEL PREDICTIONS, NOT MEASUREMENTS.

Setting (well specified, hence optimistic): the wafer is one of the literature-consistent worlds.  The mask of the author
(16 designs, `n` devices each) is measured.  Two observation models are compared, fixed in advance (not tuned):
  "counts"        per design: number of devices that switch, number that lase first along the native axis x (binomial likelihood,
                  probabilities clipped to [0.03, 0.97] so that one surprising device cannot zero a world);
  "counts+shape"  the same, plus the azimuth of the low-current state (Gaussian on the doubled angle, sd 12 deg) and the median
                  switching current of the switchers (Gaussian, sd 0.10 I/I_th; a world with no switchers where some were seen
                  is penalised like a 3-sigma miss).
The other worlds are re-weighted by the likelihood.  The design that maximises the posterior expected yield among the candidate
pool is then chosen and scored in the true world.

Reads  results/mask_predictions_per_world.npz (per-design switch / x-first probabilities for every accepted world),
       results/batch_yield_matrix.npz and results/batch_worlds.json (yield of 177 candidate designs in 64 worlds).
Writes results/value_of_first_batch.json.
Reported: (a) yield of the design chosen with no measurement, (b) after the measurement with n = 2, 4, 8 devices per design,
(c) oracle (best design of the pool in the true world); effective sample size and the recovered sign of the birefringence
coupling k_b (not identified by the literature: 25 % of the accepted worlds are positive).
Usage: python scripts/value_of_first_batch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcsel_bo import literature as L  # noqa: E402

RES = ROOT / "results"


def main(reps=20, seed=5):
    acc = L.load_anchored(RES / "anchored_worlds.json")["accepted"]
    Z = np.load(RES / "mask_predictions_per_world.npz", allow_pickle=True)
    PS, PX = Z["p_switch"], Z["p_x_first"]                       # (16, n_worlds)
    B = np.load(RES / "batch_yield_matrix.npz", allow_pickle=True)
    W = json.loads((RES / "batch_worlds.json").read_text())
    key = lambda r: (round(r["d0"], 9), round(r["b0"], 9), round(r["k_d"], 9), round(r["k_b"], 9))
    idx_of = {key(r): i for i, r in enumerate(acc)}
    wl = W["train"] + W["test"]
    gi = np.array([idx_of[key(r)] for r in wl])                   # index of each of the 64 worlds among the accepted ones
    Y = np.concatenate([B["Ytr"], B["Yte"]], axis=1)              # (177, 64)
    kb = np.array([r["k_b"] for r in wl])
    d0 = np.array([r["d0"] for r in wl])
    ps, px = np.clip(PS[:, gi], 0.03, 0.97), np.clip(PX[:, gi], 0.03, 0.97)
    ps_raw, px_raw = PS[:, gi], PX[:, gi]
    AZ, MU = Z["az_low"][:, gi], Z["mu_sw"][:, gi]
    n_w = Y.shape[1]
    rng = np.random.default_rng(seed)
    prior_choice = int(np.argmax(Y.mean(1)))
    out = dict(note="synthetic preposterior analysis, well-specified case (optimistic); model predictions", n_worlds=n_w,
               pool=int(Y.shape[0]), reps_per_world=reps)
    oracle = Y.max(0)
    out["no_measurement"] = dict(design_index=prior_choice, mean_yield=float(Y[prior_choice].mean()))
    out["oracle_mean_yield"] = float(oracle.mean())
    wrap = lambda a: (a + 90.0) % 180.0 - 90.0                 # difference of two orientations (period 180 deg)
    for model, temper in (("counts", 1.0), ("counts+shape", 1.0), ("counts+shape", 4.0)):
        for n in (2, 4, 8):
            yl, ess, sign_ok, d0err = [], [], [], []
            for j in range(n_w):
                others = np.array([i for i in range(n_w) if i != j])
                for _ in range(reps):
                    ks = rng.binomial(n, ps_raw[:, j]); kx = rng.binomial(n, px_raw[:, j])
                    ll = (binom.logpmf(ks[:, None], n, ps[:, others]) + binom.logpmf(kx[:, None], n, px[:, others])).sum(0)
                    if model == "counts+shape":
                        az_obs = AZ[:, j] + rng.normal(0, 12.0 / np.sqrt(n), 16)
                        ll = ll - 0.5 * ((wrap(AZ[:, others] - az_obs[:, None]) / 12.0) ** 2).sum(0)
                        for d in range(16):
                            if ks[d] > 0 and np.isfinite(MU[d, j]):
                                mu_obs = MU[d, j] + rng.normal(0, 0.10 / np.sqrt(ks[d]))
                                m = MU[d, others]
                                ll = ll + np.where(np.isfinite(m), -0.5 * ((m - mu_obs) / 0.10) ** 2, -4.5)
                    ll = ll / temper                           # temper > 1 flattens the weights (guards against a collapse onto 1-3 worlds)
                    w = np.exp(ll - ll.max()); w /= w.sum()
                    pick = int(np.argmax(Y[:, others] @ w))
                    yl.append(Y[pick, j]); ess.append(1.0 / np.sum(w ** 2))
                    sign_ok.append(float((w * (np.sign(kb[others]) == np.sign(kb[j]))).sum()))
                    d0err.append(abs(float((w * np.log10(d0[others])).sum()) - np.log10(d0[j])))
            out[f"{model} T{temper:g} n{n}"] = dict(observation_model=model, temperature=temper, n_devices_per_design=n, mean_yield_of_chosen_design=float(np.mean(yl)),
                                        mean_ess=float(np.mean(ess)), posterior_prob_of_true_sign_kb=float(np.mean(sign_ok)),
                                        mean_abs_error_log10_d0=float(np.mean(d0err)))
    out["prior_sign_kb_positive_fraction"] = float((kb > 0).mean())
    out["prior_std_log10_d0"] = float(np.std(np.log10(d0)))
    (RES / "value_of_first_batch.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
