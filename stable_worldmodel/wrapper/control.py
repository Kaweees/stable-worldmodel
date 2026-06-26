"""Control wrappers for environment-side action semantics."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each requested action for multiple primitive env steps.

    The wrapped environment still exposes a normal Gymnasium interface: one
    call to ``step()`` consumes one policy/decision action. Internally, that
    action is executed up to ``repeat`` times, rewards are summed, and the
    wrapper returns the final observation reached. Repetition stops early if the
    inner environment terminates or truncates.
    """

    def __init__(self, env: gym.Env, repeat: int = 1):
        """Initialize the wrapper.

        Args:
            env: Environment to wrap.
            repeat: Maximum number of primitive ``env.step`` calls per action.

        Raises:
            ValueError: If ``repeat`` is not a positive integer.
        """
        try:
            repeat_int = int(repeat)
        except (TypeError, ValueError) as exc:
            raise ValueError('repeat must be a positive integer.') from exc
        if repeat_int != repeat or repeat_int < 1:
            raise ValueError('repeat must be a positive integer.')
        super().__init__(env)
        self.repeat = repeat_int
        self.action_repeat = repeat_int

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        """Reset the env and expose reset-safe ``repeat_count`` metadata."""
        obs, info = self.env.reset(*args, **kwargs)
        info = dict(info)
        info['repeat_count'] = np.int32(0)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Execute ``action`` up to ``repeat`` times."""
        total_reward = 0.0
        repeat_count = 0
        terminated = False
        truncated = False
        obs = None
        info: dict[str, Any] = {}

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, step_info = self.env.step(
                action
            )
            info = dict(step_info)
            total_reward += 0.0 if reward is None else float(reward)
            repeat_count += int(np.asarray(info.get('repeat_count', 1)))
            if terminated or truncated:
                break

        info['repeat_count'] = np.int32(repeat_count)
        return obs, total_reward, bool(terminated), bool(truncated), info
