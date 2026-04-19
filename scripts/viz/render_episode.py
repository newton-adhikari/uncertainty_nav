# Renders a clean matplotlib animation showing:
# Robot trajectory color-coded by epistemic uncertainty (green→red)

from uncertainty_nav.models import PolicyNetwork, DeepEnsemble
from uncertainty_nav.baselines import VanillaMLP
from uncertainty_nav.nav_env import PartialObsNavEnv, ENV_A, ENV_B, ENV_C, ENV_D, EnvConfig

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "experiments/videos"
ENV_MAP = {"A": ENV_A, "B": ENV_B, "C": ENV_C, "D": ENV_D}

# ── Color utilities ──────────────────────────────────────────────────

def uncertainty_color(u, u_max=1.0):
    """Map uncertainty to green→yellow→red."""
    t = min(u / u_max, 1.0)
    if t < 0.5:
        return (2 * t, 1.0, 0.0, 0.9)       # green → yellow
    else:
        return (1.0, 2 * (1 - t), 0.0, 0.9)  # yellow → red
