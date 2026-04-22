#!/usr/bin/env bash
# ==============================================================================
# Single script that regenerates every JSON, figure, and metric.
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EVAL="python3 scripts/eval/evaluate.py"
N_EP=200
N_SEEDS=5
OUTDIR="experiments/results"
PLOTDIR="experiments/plots"
AUDIT_LOG="reproduce_audit.txt"

# Clean previous results
rm -f "$OUTDIR"/*.json
rm -f "$OUTDIR/noise_sweep"/*.json 2>/dev/null || true
rm -f "$PLOTDIR"/*.pdf "$PLOTDIR"/*.png 2>/dev/null || true
mkdir -p "$OUTDIR" "$OUTDIR/noise_sweep" "$PLOTDIR"

echo "============================================================"
echo " Output:  $OUTDIR/"
echo " Plots:   $PLOTDIR/"
echo " Audit:   $AUDIT_LOG"
echo " Episodes: $N_EP   Seeds: $N_SEEDS"
echo "============================================================"
echo ""

# ==================================================================
# STEP 1: Baselines (no uncertainty) × 4 environments
# ==================================================================
echo ">>> [1/10] Baselines × 4 environments"
for method in vanilla lstm gru large_mlp; do
    for env in A B C D; do
        echo "  $method on Env $env"
        $EVAL --policy "$method" --env "$env" \
            --n_episodes $N_EP --n_seeds $N_SEEDS
    done
done

# ==================================================================
# STEP 2: Deep Ensemble (N=5) × 4 environments
# ==================================================================
echo ""
echo ">>> [2/10] Ensemble (N=5) × 4 environments"
for env in A B C D; do
    echo "  ensemble on Env $env"
    $EVAL --policy ensemble --env "$env" \
        --n_episodes $N_EP --n_seeds $N_SEEDS --n_members 5
done

# ==================================================================
# STEP 3: MC-Dropout × 4 environments × T={5,10,20}
#   Output: mc_dropout_T{T}_env{X}.json (unique per T)
# ==================================================================
echo ""
echo ">>> [3/10] MC-Dropout × 4 environments × T={5,10,20}"
for T in 5 10 20; do
    for env in A B C D; do
        echo "  mc_dropout T=$T on Env $env"
        $EVAL --policy mc_dropout --env "$env" \
            --mc_samples $T \
            --n_episodes $N_EP --n_seeds $N_SEEDS
    done
done

# ==================================================================
# STEP 4: Ensemble size ablation (N=1,2,3,5,10) on Env B
# ==================================================================
echo ""
echo ">>> [4/10] Ensemble size ablation (Env B, 500 episodes)"
for N in 1 2 3 5 10; do
    echo "  ensemble N=$N on Env B"
    $EVAL --policy ensemble --env B \
        --n_episodes 500 --n_seeds 5 --n_members $N
done

# ==================================================================
# STEP 5: Additional ablations (threshold, capacity, action modulation)
# ==================================================================
echo ""
echo ">>> [5/10] Ablation studies"
python3 scripts/ablation/run_ablations.py

# ==================================================================
# STEP 6: Ensemble size AUROC (generates ensemble_size_auroc.json for fig5)
# ==================================================================
echo ""
echo ">>> [6/10] Ensemble size AUROC ablation"
python3 scripts/eval/evaluate_all_envs.py --auroc_ablation

# ==================================================================
# STEP 7: Noise sweep
# ==================================================================
echo ""
echo ">>> [7/10] Noise sweep (Env A layout)"
for method in ensemble vanilla large_mlp mc_dropout; do
    echo "  $method noise sweep"
    if [ "$method" = "mc_dropout" ]; then
        $EVAL --policy "$method" --env A \
            --mc_samples 20 \
            --n_episodes $N_EP --n_seeds $N_SEEDS --noise_sweep \
            --output_dir "$OUTDIR/noise_sweep"
    else
        $EVAL --policy "$method" --env A \
            --n_episodes $N_EP --n_seeds $N_SEEDS --noise_sweep \
            --output_dir "$OUTDIR/noise_sweep"
    fi
done

# ==================================================================
# STEP 8: Post-hoc analysis
# ==================================================================
echo ""
echo ">>> [8/10] Post-hoc analysis"

echo "  Temperature scaling..."
python3 scripts/eval/compute_temperature_scaling.py \
    --policy mc_dropout --envs A B C D --n_episodes 500

echo "  Cosine similarity..."
python3 scripts/eval/compute_cosine_similarity.py \
    --envs A B C D --n_episodes 200

echo "  Inference latency..."
python3 scripts/eval/measure_inference_latency.py --device cpu --n_trials 500

echo "  Bootstrap AUROC confidence intervals..."
python3 scripts/eval/compute_auroc_ci.py

# ==================================================================
# STEP 9: Generate all figures
# ==================================================================
echo ""
echo ">>> [9/10] Generating figures"
python3 scripts/eval/plot_results.py


# ==================================================================
# STEP 10: Full audit — verify every number against tables
# ==================================================================
echo ""
echo ">>> [10/10] Audit"
echo ""

# Write audit to both stdout and file
python3 - "$OUTDIR" <<'PYEOF' | tee "$AUDIT_LOG"
import json, os, sys, glob

OUTDIR = sys.argv[1]
files = sorted(glob.glob(f'{OUTDIR}/*.json'))
print(f'Found {len(files)} result files')
print()

# ── Table III: Main results ──────────────────────────────────────
print('=' * 70)
print('TABLE III — MAIN RESULTS (SR ± std, AUROC, ECE)')
print('=' * 70)
for pol in ['vanilla','lstm','gru','large_mlp','mc_dropout','ensemble']:
    for env in ['A','B','C','D']:
        if pol == 'mc_dropout':
            candidates = [f'{OUTDIR}/mc_dropout_T20_env{env}.json']
        else:
            candidates = [f'{OUTDIR}/{pol}_env{env}.json']
        for path in candidates:
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                sr = d['success_rate']
                std = d.get('success_rate_std', 0)
                spl = d.get('mean_spl', 0)
                cr = d.get('collision_rate', 0)
                tr = d.get('timeout_rate', 0)
                unc = d.get('mean_uncertainty', 0)
                auroc = d.get('auroc_failure', '-')
                ece = d.get('ece', '-')
                r = d.get('uncertainty_success_correlation', '-')
                if isinstance(auroc, float): auroc = f'{auroc:.3f}'
                if isinstance(ece, float): ece = f'{ece:.3f}'
                if isinstance(r, float): r = f'{r:.3f}'
                print(f'  {pol:14s} Env {env}: SR={sr:.3f}±{std:.3f} SPL={spl:.3f} '
                      f'CR={cr:.3f} TR={tr:.3f} unc={unc:.4f} AUROC={auroc} ECE={ece} r={r}')
                break
        else:
            print(f'  {pol:14s} Env {env}: MISSING')
    print()

# ── Table IV: AUROC / ECE / r ────────────────────────────────────
print('=' * 70)
print('TABLE IV — FAILURE PREDICTION (AUROC, ECE, r)')
print('=' * 70)
for pol, label in [('ensemble','Ensemble'), ('mc_dropout','MC-Drop T=20')]:
    for env in ['A','C','D','B']:
        if pol == 'mc_dropout':
            path = f'{OUTDIR}/mc_dropout_T20_env{env}.json'
        else:
            path = f'{OUTDIR}/{pol}_env{env}.json'
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            auroc = d.get('auroc_failure', '-')
            ece = d.get('ece', '-')
            r = d.get('uncertainty_success_correlation', '-')
            if isinstance(auroc, float): auroc = f'{auroc:.3f}'
            if isinstance(ece, float): ece = f'{ece:.3f}'
            if isinstance(r, float): r = f'{r:.3f}'
            print(f'  {label:14s} Env {env}: AUROC={auroc}  ECE={ece}  r={r}')
    print()

# ── Table V: Routing ─────────────────────────────────────────────
print('=' * 70)
print('TABLE V — ROUTING (Env B)')
print('=' * 70)
for label, path in [
    ('Ensemble', f'{OUTDIR}/ensemble_envB.json'),
    ('MC-Drop T=20', f'{OUTDIR}/mc_dropout_T20_envB.json'),
]:
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f'  {label}:')
        for r in d.get('routing_table', []):
            print(f'    {r["strategy"]:30s}  auto_SR={r["autonomous_sr"]:.3f}  human={r["human_burden"]:.0%}')
    print()

# ── Table VI: T ablation ─────────────────────────────────────────
print('=' * 70)
print('TABLE VI — MC-DROPOUT T ABLATION (Env B)')
print('=' * 70)
for T in [5, 10, 20]:
    path = f'{OUTDIR}/mc_dropout_T{T}_envB.json'
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f'  T={T:2d}: SR={d["success_rate"]:.3f}±{d.get("success_rate_std",0):.3f}  '
              f'AUROC={d.get("auroc_failure",0):.3f}  ECE={d.get("ece",0):.3f}')
print()

# ── Table VII: Ensemble size AUROC ────────────────────────────────
print('=' * 70)
print('TABLE VII — ENSEMBLE SIZE ABLATION (Env B)')
print('=' * 70)
auroc_path = f'{OUTDIR}/ensemble_size_auroc.json'
if os.path.exists(auroc_path):
    with open(auroc_path) as f:
        data = json.load(f)
    for key in ['N=1','N=2','N=3','N=5','N=10']:
        if key in data:
            d = data[key]
            print(f'  {key}: SR={d["success_rate"]:.3f}  AUROC={d["auroc"]:.3f}  '
                  f'ECE={d["ece"]:.3f}  unc={d["mean_uncertainty"]:.4f}')
else:
    print('  MISSING — run: python3 scripts/eval/evaluate_all_envs.py --auroc_ablation')
print()

# ── Uncertainty ratio ─────────────────────────────────────────────
print('=' * 70)
print('UNCERTAINTY RATIO (Env D / Env C)')
print('=' * 70)
for pol, label in [('ensemble', 'Ensemble'), ('mc_dropout', 'MC-Dropout')]:
    c_path = f'{OUTDIR}/{pol}_T20_envC.json' if pol == 'mc_dropout' else f'{OUTDIR}/{pol}_envC.json'
    d_path = f'{OUTDIR}/{pol}_T20_envD.json' if pol == 'mc_dropout' else f'{OUTDIR}/{pol}_envD.json'
    unc_c = unc_d = None
    if os.path.exists(c_path):
        with open(c_path) as f: unc_c = json.load(f).get('mean_uncertainty', 0)
    if os.path.exists(d_path):
        with open(d_path) as f: unc_d = json.load(f).get('mean_uncertainty', 0)
    if unc_c and unc_d and unc_c > 0:
        print(f'  {label}: D={unc_d:.4f} / C={unc_c:.4f} = {unc_d/unc_c:.1f}x')
    else:
        print(f'  {label}: data missing')
print()

# ── Temperature scaling ───────────────────────────────────────────
print('=' * 70)
print('TEMPERATURE SCALING')
print('=' * 70)
ts_path = f'{OUTDIR}/temperature_scaling_mc_dropout.json'
if os.path.exists(ts_path):
    with open(ts_path) as f: ts = json.load(f)
    for env in ['A','B','C','D']:
        if env in ts:
            d = ts[env]
            print(f'  Env {env}: raw={d["raw_ece"]:.4f} -> cal={d["calibrated_ece"]:.4f} '
                  f'(T={d["optimal_temperature"]:.3f}, reduction={d["ece_reduction_pct"]:.1f}%)')
else:
    print('  MISSING')
print()

# ── Cosine similarity ─────────────────────────────────────────────
print('=' * 70)
print('COSINE SIMILARITY')
print('=' * 70)
cs_path = f'{OUTDIR}/ensemble_cosine_similarity.json'
if os.path.exists(cs_path):
    with open(cs_path) as f: cs = json.load(f)
    for env in ['A','B','C','D']:
        if env in cs:
            d = cs[env]
            s = f'{d["success_mean_cosine"]:.4f}' if d.get("success_mean_cosine") else 'N/A'
            fl = f'{d["failure_mean_cosine"]:.4f}' if d.get("failure_mean_cosine") else 'N/A'
            print(f'  Env {env}: overall={d["mean_cosine_similarity"]:.4f}  success={s}  failure={fl}')
else:
    print('  MISSING')
print()

# ── Inference latency ─────────────────────────────────────────────
print('=' * 70)
print('INFERENCE LATENCY')
print('=' * 70)
lat_path = f'{OUTDIR}/inference_latency_cpu.json'
if os.path.exists(lat_path):
    with open(lat_path) as f: lat = json.load(f)
    mc = lat.get('mc_dropout', {})
    for T in ['T=5','T=10','T=20']:
        if T in mc:
            print(f'  MC-Drop {T}: mean={mc[T]["mean_ms"]:.3f}ms  p95={mc[T]["p95_ms"]:.3f}ms')
    ens = lat.get('ensemble_N5', {})
    if 'mean_ms' in ens:
        print(f'  Ensemble N=5: mean={ens["mean_ms"]:.3f}ms  p95={ens["p95_ms"]:.3f}ms')
else:
    print('  MISSING')
print()

# ── File manifest ─────────────────────────────────────────────────
print('=' * 70)
print(f'FILE MANIFEST ({len(files)} files)')
print('=' * 70)
for f in files:
    size = os.path.getsize(f)
    print(f'  {f:55s} {size:>6d} bytes')
PYEOF

echo ""
echo "============================================================"
echo " DONE"
echo " Results: $OUTDIR/"
echo " Plots:   $PLOTDIR/"
echo " Audit:   $AUDIT_LOG"
echo "============================================================"
