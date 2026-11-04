# Autonomous Lunar Lander: PPO and Reward Shaping

Train a neural-network agent to land the Gymnasium LunarLander-v3 spacecraft using Proximal Policy Optimization (PPO), then study how reward design changes learned behavior.

## Project goals

1. Train a baseline PPO agent using the environment's original reward.
2. Compare a deliberately flawed low-vertical-speed bonus with the baseline.
3. Test potential-based reward shaping.
4. Evaluate every policy using the **original, unmodified environment reward**.
5. Save evaluation CSVs, comparison charts, checkpoints, and optional videos.

## Architecture

- Environment: Gymnasium `LunarLander-v3` (Box2D)
- Algorithm: PPO (Stable-Baselines3)
- Policy: MLP Actor-Critic; actor and critic each use two 64-unit hidden layers
- Framework: PyTorch through Stable-Baselines3

The actor outputs a distribution over four discrete actions. The critic estimates state value. PPO updates the policy and value networks using rollout data.

## Windows setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Box2D installation fails with a SWIG-related error, install SWIG and retry:

```powershell
pip install swig
pip install "gymnasium[box2d]"
```

## Train

From the repository root:

```powershell
python training/train.py --mode baseline --timesteps 300000
python training/train.py --mode hover --timesteps 100000
python training/train.py --mode shaped --timesteps 100000
```

For a short smoke test first:

```powershell
python training/train.py --mode baseline --timesteps 10000
```

Models and checkpoints are saved under `models/`. Training logs are under `logs/`.

## Evaluate

Evaluation always uses the original Gymnasium reward, not the custom training reward:

```powershell
python evaluation/evaluate.py --model lunar_lander_baseline --episodes 20
python evaluation/evaluate.py --model lunar_lander_baseline --episodes 5 --record
```

For the other runs, replace the model name with `lunar_lander_hover` or `lunar_lander_shaped`.

## Compare

```powershell
python visualization/plot_results.py
```

This creates `results/comparison_summary.csv` and `results/evaluation_comparison.png`.

## Reward experiments

- `baseline`: original Gymnasium reward.
- `hover`: adds +1 whenever absolute vertical velocity is below 0.1. This is intentionally flawed because it rewards slow vertical movement without requiring safe landing. It may or may not produce hovering; inspect trajectories and episode outcomes rather than assuming.
- `shaped`: adds the potential-based term `gamma * Phi(next_state) - Phi(state)`. The potential uses position, velocity, angle, angular velocity, and leg contact. Terminal next-state potential is set to zero.

Potential-based shaping is theoretically policy-invariant under the usual assumptions when applied consistently. The chosen potential is a heuristic; it does not guarantee faster learning or a successful landing.

## Evaluation protocol

Use at least 20 episodes for a quick comparison and preferably 100 or more for stronger evidence. Report:
- Mean and standard deviation of original episode reward
- Fraction of episodes scoring at least 200 (Gymnasium's commonly used solved threshold)
- Mean episode length
- Video inspection of behavior

The short training run is not proof of a solved task. Training is stochastic; increase timesteps and repeat seeds for reliable claims.


