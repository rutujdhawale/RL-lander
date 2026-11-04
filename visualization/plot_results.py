from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from configs.config import RESULTS_DIR


def main():
    files = sorted(RESULTS_DIR.glob("*_evaluation.csv"))
    if not files:
        print("No evaluation CSVs found. Run evaluation.py first.")
        return

    frames = []
    for file in files:
        df = pd.read_csv(file)
        df["model"] = file.name.replace("_evaluation.csv", "")
        frames.append(df)
    all_results = pd.concat(frames, ignore_index=True)

    summary = all_results.groupby("model").agg(
        mean_reward=("original_reward", "mean"),
        std_reward=("original_reward", "std"),
        success_rate=("success_200_threshold", "mean"),
        mean_length=("episode_length", "mean"),
    ).reset_index()
    summary["success_rate"] *= 100
    summary.to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)
    print(summary.to_string(index=False))

    plt.figure(figsize=(10, 5))
    for model, group in all_results.groupby("model"):
        plt.plot(group["episode"], group["original_reward"], marker="o", label=model)
    plt.axhline(200, linestyle="--", label="200-point solved threshold")
    plt.xlabel("Evaluation episode")
    plt.ylabel("Original environment reward")
    plt.title("LunarLander: Evaluation reward by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "evaluation_comparison.png", dpi=160)
    print(f"Saved plots and summary in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
