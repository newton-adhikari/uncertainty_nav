# this script
# compute bootstrap confidence intervals 
# on AUROC for the headline comparison.

import json, numpy as np

def bootstrap_auroc(uncertainties, failures, n_bootstrap=1000, ci=0.95):
    rng = np.random.default_rng(42)
    n = len(uncertainties)
    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        u, f = uncertainties[idx], failures[idx]
        if len(np.unique(f)) < 2:
            continue
        pos, neg = u[f == 1], u[f == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        auroc = np.mean([float(p > n_) + 0.5 * float(p == n_)
                         for p in pos for n_ in neg])
        aurocs.append(auroc)
    aurocs = np.sort(aurocs)
    lo = aurocs[int((1 - ci) / 2 * len(aurocs))]
    hi = aurocs[int((1 + ci) / 2 * len(aurocs))]
    return float(lo), float(hi), float(np.mean(aurocs))

for method, label in [("mc_dropout_T20", "MC-Dropout T=20"), ("ensemble", "Ensemble")]:
    path = f"experiments/results/{method}_envB.json"
    with open(path) as f:
        d = json.load(f)
    cal = d["uncertainty_calibration"]

    # Reconstruct per-episode data from quartiles