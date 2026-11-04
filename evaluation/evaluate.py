import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from configs.config import ENV_ID, RESULTS_DIR, VIDEOS_DIR


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the ORIGINAL reward.")
    parser.add_argument("--model", default="lunar_lander_baseline",
                        help="Model filename without .zip, relative to models/.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    model_path = ROOT / "models" / args.model
    if not Path(str(model_path) + ".zip").exists() and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}.zip. Train it first.")

    env = gym.make(ENV_ID, render_mode="rgb_array" if args.record else None)
    if args.record:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(VIDEOS_DIR / args.model),
            episode_trigger=lambda episode_id: episode_id < min(args.episodes, 5),
            name_prefix=args.model,
        )

    model = PPO.load(str(model_path))
    rows = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=10_000 + episode)
        terminated = truncated = False
        total_reward = 0.0
        length = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)  # Original environment reward only
            length += 1

        rows.append({
            "episode": episode + 1,
            "original_reward": total_reward,
            "episode_length": length,
            "success_200_threshold": total_reward >= 200,
            "terminated": terminated,
            "truncated": truncated,
        })
        print(f"Episode {episode + 1:03d}: reward={total_reward:8.2f}, "
              f"steps={length:4d}, success_200={total_reward >= 200}")

    env.close()
    df = pd.DataFrame(rows)
    output = RESULTS_DIR / f"{args.model}_evaluation.csv"
    df.to_csv(output, index=False)

    print("\n===== EVALUATION SUMMARY (original reward) =====")
    print(f"Episodes: {len(df)}")
    print(f"Mean reward: {df.original_reward.mean():.2f}")
    print(f"Std reward: {df.original_reward.std(ddof=0):.2f}")
    print(f"Success rate (>=200): {100 * df.success_200_threshold.mean():.1f}%")
    print(f"Mean episode length: {df.episode_length.mean():.1f}")
    print(f"Saved results: {output}")


if __name__ == "__main__":
    main()
