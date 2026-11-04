from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"
VIDEOS_DIR = ROOT / "videos"

for directory in (MODELS_DIR, LOGS_DIR, RESULTS_DIR, VIDEOS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ENV_ID = "LunarLander-v3"
SEED = 42

# A reasonable starting point. Increase to 500_000–1_000_000 for stronger results.
DEFAULT_TIMESTEPS = 300_000

PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "verbose": 1,
    "policy_kwargs": {
        "net_arch": {"pi": [64, 64], "vf": [64, 64]}
    },
}
