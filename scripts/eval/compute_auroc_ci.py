# this script
# compute bootstrap confidence intervals 
# on AUROC for the headline comparison.

import json, numpy as np, os

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

results = {}

for method, label in [("mc_dropout_T20", "MC-Dropout T=20"), ("ensemble", "Ensemble")]:
    path = f"experiments/results/{method}_envB.json"
    if not os.path.exists(path):
        print(f"{label}: MISSING {path}")
        continue
    with open(path) as f:
        d = json.load(f)

    if "per_episode_uncertainty" in d and "per_episode_success" in d:
        uncs = np.array(d["per_episode_uncertainty"])
        successes = np.array(d["per_episode_success"], dtype=float)
        failures = 1.0 - successes
        source = "per-episode data"
    else:
        print(f"{label}: WARNING — no per-episode arrays found, skipping")
        continue

    lo, hi, mean = bootstrap_auroc(uncs, failures)
    results[method] = {"auroc": d["auroc_failure"], "ci_lo": lo, "ci_hi": hi, "mean": mean}
    print(f"{label}: AUROC={d['auroc_failure']:.3f}, bootstrap mean={mean:.3f}, "
          f"95% CI=[{lo:.3f}, {hi:.3f}] (from {source}, n={len(uncs)})")

# Save
out_path = "experiments/results/auroc_bootstrap_ci.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
