import gymnasium as gym


def potential(observation):
    """
    Bounded-ish state potential: higher is better.
    LunarLander observations: x, y, vx, vy, angle, angular_velocity, leg1, leg2.
    This is a shaping heuristic, not a complete landing controller.
    """
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation
    distance_and_motion = (
        1.0 * abs(float(x))
        + 0.35 * abs(float(y))
        + 0.5 * abs(float(vx))
        + 0.35 * abs(float(vy))
        + 0.5 * abs(float(angle))
        + 0.15 * abs(float(angular_velocity))
    )
    leg_bonus = 0.25 * (float(left_leg) + float(right_leg))
    return -distance_and_motion + leg_bonus


class RewardExperimentWrapper(gym.Wrapper):
    """
    mode:
      baseline: pass through the original Gymnasium reward unchanged.
      hover: deliberately flawed bonus for low vertical speed.
      shaped: potential-based shaping F = gamma * Phi(s') - Phi(s).

    The hover bonus is intentionally capable of rewarding slow vertical motion
    without requiring a landing. Whether it actually learns to hover must be
    established by evaluating and inspecting trained behavior.
    """
    def __init__(self, env, mode="baseline", gamma=0.99, hover_bonus=1.0):
        super().__init__(env)
        if mode not in {"baseline", "hover", "shaped"}:
            raise ValueError(f"Unknown reward mode: {mode}")
        self.mode = mode
        self.gamma = gamma
        self.hover_bonus = hover_bonus
        self.previous_potential = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_potential = potential(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(reward)

        if self.mode == "hover":
            vertical_speed = abs(float(observation[3]))
            if vertical_speed < 0.1:
                shaped_reward += self.hover_bonus

        elif self.mode == "shaped":
            # For terminal transitions set Phi(s_terminal)=0.
            next_potential = 0.0 if (terminated or truncated) else potential(observation)
            shaping = self.gamma * next_potential - self.previous_potential
            shaped_reward += shaping
            self.previous_potential = next_potential

        info = dict(info)
        info["base_reward"] = float(reward)
        info["training_reward"] = float(shaped_reward)
        return observation, shaped_reward, terminated, truncated, info
