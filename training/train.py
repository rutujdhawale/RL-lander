import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from configs.config import ENV_ID, SEED, DEFAULT_TIMESTEPS, PPO_KWARGS, MODELS_DIR, LOGS_DIR
from environments.reward_wrapper import RewardExperimentWrapper


def main():
    parser = argparse.ArgumentParser(description="Train a PPO LunarLander agent.")
    parser.add_argument("--mode", choices=["baseline", "hover", "shaped"], default="baseline")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--tag", default=None, help="Optional model name suffix.")
    args = parser.parse_args()

    run_name = args.tag or args.mode
    model_path = MODELS_DIR / f"lunar_lander_{run_name}"
    log_path = LOGS_DIR / run_name
    checkpoint_dir = MODELS_DIR / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    env = RewardExperimentWrapper(env, mode=args.mode)
    env = Monitor(env)

    callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_lander",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        tensorboard_log=str(LOGS_DIR / "tensorboard"),
        **PPO_KWARGS,
    )

    print(f"\nTraining mode: {args.mode}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Saving model to: {model_path}.zip\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name=run_name,
        progress_bar=True,
    )
    model.save(str(model_path))
    env.close()
    print(f"\nTraining complete. Saved: {model_path}.zip")


if __name__ == "__main__":
    main()
