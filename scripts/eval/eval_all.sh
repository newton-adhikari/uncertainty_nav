#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EVAL="python3 scripts/eval/evaluate.py"
N_EP=200
N_SEEDS=5
OUTDIR="experiments/results"
mkdir -p "$OUTDIR"

echo "============================================================"
echo " Output: $OUTDIR/"
echo " Episodes: $N_EP  Seeds: $N_SEEDS"
echo "============================================================"

# 1. ALL BASELINES (no uncertainty) × ALL ENVIRONMENTS

echo ""
echo ">>> Step 1: Baselines × 4 environments"
for method in vanilla lstm gru large_mlp; do
    for env in A B C D; do
        echo "  $method on Env $env ..."
        $EVAL --policy "$method" --env "$env" \
            --n_episodes $N_EP --n_seeds $N_SEEDS
    done
done

# 2. ENSEMBLE × ALL ENVIRONMENTS

echo ""
echo ">>> Step 2: Ensemble (N=5) × 4 environments"
for env in A B C D; do
    echo "  ensemble on Env $env ..."
    $EVAL --policy ensemble --env "$env" \
        --n_episodes $N_EP --n_seeds $N_SEEDS --n_members 5
done

# ------------------------------------------------------------------
# 3. MC-DROPOUT × ALL ENVIRONMENTS × MULTIPLE T VALUES
#    Each T gets its own filename: mc_dropout_T{T}_env{X}.json
# ------------------------------------------------------------------
echo ""
echo ">>> Step 3: MC-Dropout × 4 environments × T={5,10,20}"
for T in 5 10 20; do
    for env in A B C D; do
        echo "  mc_dropout T=$T on Env $env ..."
        $EVAL --policy mc_dropout --env "$env" \
            --mc_samples $T \
            --n_episodes $N_EP --n_seeds $N_SEEDS
    done
done

# ------------------------------------------------------------------
# 4. ENSEMBLE SIZE ABLATION (N=1,2,3,5,10) on Env B
# ------------------------------------------------------------------
echo ""
echo ">>> Step 4: Ensemble size ablation on Env B"
for N in 1 2 3 5 10; do
    echo "  ensemble N=$N on Env B ..."
    $EVAL --policy ensemble --env B \
        --n_episodes 500 --n_seeds 5 --n_members $N
done

# ------------------------------------------------------------------
# 5. ABLATION STUDIES (threshold, capacity, action modulation)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 5: Ablation studies"
python3 scripts/ablation/run_ablations.py

# ------------------------------------------------------------------
# 6. NOISE SWEEP (robustness curve on Env A layout)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 6: Noise sweep (Env A layout)"
for method in ensemble vanilla large_mlp; do
    echo "  $method noise sweep ..."
    $EVAL --policy "$method" --env A \
        --n_episodes $N_EP --n_seeds $N_SEEDS --noise_sweep
done

# ------------------------------------------------------------------
# 7. POST-HOC ANALYSIS (temperature scaling, cosine similarity, latency)
# ------------------------------------------------------------------
echo ""
echo ">>> Step 7: Post-hoc analysis scripts"

echo "  Temperature scaling (calibrated ECE)..."
python3 scripts/eval/compute_temperature_scaling.py \
    --policy mc_dropout --envs A B C D --n_episodes 500

echo "  Ensemble cosine similarity (diversity collapse)..."
python3 scripts/eval/compute_cosine_similarity.py \
    --envs A B C D --n_episodes 200

echo "  Inference latency benchmark..."
python3 scripts/eval/measure_inference_latency.py --device cpu --n_trials 500

# ------------------------------------------------------------------
# 8. GENERATE PLOTS
# ------------------------------------------------------------------
echo ""
echo ">>> Step 8: Generating plots"
if [ -f scripts/eval/plot_results.py ]; then
    python3 scripts/eval/plot_results.py
fi

# ------------------------------------------------------------------
# 9. AUDIT: Print all results for paper cross-check
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo " AUDIT: All results for paper verification"
echo "============================================================"
python3 -c "
import json, os, glob

files = sorted(glob.glob('$OUTDIR/*.json'))
print(f'Found {len(files)} result files\n')

# Main table numbers
print('=== MAIN TABLE (SR ± std) ===')
for pol in ['vanilla','lstm','gru','large_mlp','mc_dropout','ensemble']:
    for env in ['A','B','C','D']:
        # Try T=20 for mc_dropout first, then plain
        if pol == 'mc_dropout':
            candidates = [
                f'$OUTDIR/mc_dropout_T20_env{env}.json',
                f'$OUTDIR/mc_dropout_env{env}.json',
            ]
        else:
            candidates = [f'$OUTDIR/{pol}_env{env}.json']
        found = False
        for path in candidates:
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                sr = d['success_rate']
                std = d.get('success_rate_std', 0)
                auroc = d.get('auroc_failure', '-')
                ece = d.get('ece', '-')
                if isinstance(auroc, float):
                    auroc = f'{auroc:.3f}'
                if isinstance(ece, float):
                    ece = f'{ece:.3f}'
                print(f'  {pol:14s} Env {env}: SR={sr:.3f}±{std:.3f}  AUROC={auroc}  ECE={ece}')
                found = True
                break
        if not found:
            print(f'  {pol:14s} Env {env}: MISSING')
    print()

# MC-Dropout T comparison
print('=== MC-DROPOUT T COMPARISON (Env B) ===')
for T in [5, 10, 20]:
    path = f'$OUTDIR/mc_dropout_T{T}_envB.json'
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f'  T={T:2d}: SR={d[\"success_rate\"]:.3f}  AUROC={d.get(\"auroc_failure\",0):.3f}  ECE={d.get(\"ece\",0):.3f}')
    else:
        print(f'  T={T:2d}: MISSING')

# Routing table
print('\n=== ROUTING TABLE (Env B) ===')
for pol_label, path_candidates in [
    ('Ensemble', [f'$OUTDIR/ensemble_envB.json']),
    ('MC-Drop T=20', [f'$OUTDIR/mc_dropout_T20_envB.json', f'$OUTDIR/mc_dropout_envB.json']),
]:
    for path in path_candidates:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            rt = d.get('routing_table', [])
            print(f'  {pol_label}:')
            for r in rt:
                print(f'    {r[\"strategy\"]:30s}  auto_SR={r[\"autonomous_sr\"]:.3f}  human={r[\"human_burden\"]:.0%}')
            break

# Uncertainty ratio (sensor vs layout shift)
print('\n=== UNCERTAINTY RATIO (Env D / Env C) ===')
for pol, label in [('ensemble', 'Ensemble'), ('mc_dropout', 'MC-Dropout')]:
    paths_c = [f'$OUTDIR/{pol}_T20_envC.json', f'$OUTDIR/{pol}_envC.json']
    paths_d = [f'$OUTDIR/{pol}_T20_envD.json', f'$OUTDIR/{pol}_envD.json']
    unc_c = unc_d = None
    for p in paths_c:
        if os.path.exists(p):
            with open(p) as f:
                unc_c = json.load(f).get('mean_uncertainty', 0)
            break
    for p in paths_d:
        if os.path.exists(p):
            with open(p) as f:
                unc_d = json.load(f).get('mean_uncertainty', 0)
            break
    if unc_c and unc_d and unc_c > 0:
        ratio = unc_d / unc_c
        print(f'  {label}: Env D unc={unc_d:.4f} / Env C unc={unc_c:.4f} = {ratio:.1f}x')
    else:
        print(f'  {label}: data missing')

print('\n=== FILE LIST ===')
for f in files:
    print(f'  {f}')

# Post-hoc analysis results
print('\n=== TEMPERATURE SCALING ===')
ts_path = f'$OUTDIR/temperature_scaling_mc_dropout.json'
if os.path.exists(ts_path):
    with open(ts_path) as f:
        ts = json.load(f)
    for env in ['A','B','C','D']:
        if env in ts:
            d = ts[env]
            print(f'  Env {env}: raw_ECE={d[\"raw_ece\"]:.4f} -> cal_ECE={d[\"calibrated_ece\"]:.4f} (T={d[\"optimal_temperature\"]:.3f})')
else:
    print('  MISSING')

print('\n=== COSINE SIMILARITY ===')
cs_path = f'$OUTDIR/ensemble_cosine_similarity.json'
if os.path.exists(cs_path):
    with open(cs_path) as f:
        cs = json.load(f)
    for env in ['A','B','C','D']:
        if env in cs:
            d = cs[env]
            s = d.get('success_mean_cosine', 'N/A')
            fl = d.get('failure_mean_cosine', 'N/A')
            if isinstance(s, float): s = f'{s:.4f}'
            if isinstance(fl, float): fl = f'{fl:.4f}'
            print(f'  Env {env}: overall={d[\"mean_cosine_similarity\"]:.4f} success={s} failure={fl}')
else:
    print('  MISSING')

print('\n=== INFERENCE LATENCY ===')
lat_path = f'$OUTDIR/inference_latency_cpu.json'
if os.path.exists(lat_path):
    with open(lat_path) as f:
        lat = json.load(f)
    mc = lat.get('mc_dropout', {})
    for T in ['T=5','T=10','T=20']:
        if T in mc:
            print(f'  MC-Drop {T}: {mc[T][\"mean_ms\"]:.3f}ms (p95={mc[T][\"p95_ms\"]:.3f}ms)')
    ens = lat.get('ensemble_N5', {})
    if 'mean_ms' in ens:
        print(f'  Ensemble N=5: {ens[\"mean_ms\"]:.3f}ms')
else:
    print('  MISSING')
"

echo ""
echo "============================================================"
echo " DONE. All results in $OUTDIR/"
echo " Cross-check the AUDIT output above against paper tables."
echo "============================================================"