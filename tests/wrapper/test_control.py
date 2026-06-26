import gymnasium as gym
import numpy as np
import pytest

from stable_worldmodel.world.env_pool import EnvPool
from stable_worldmodel.wrapper import ActionRepeatWrapper


class CountingEnv(gym.Env):
    def __init__(self, terminated_at=None, truncated_at=None):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.terminated_at = terminated_at
        self.truncated_at = truncated_at
        self.step_count = 0
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.actions = []
        return self._obs(), {'raw_step': np.int32(self.step_count)}

    def step(self, action):
        self.step_count += 1
        self.actions.append(np.asarray(action).copy())
        terminated = self.terminated_at == self.step_count
        truncated = self.truncated_at == self.step_count
        return (
            self._obs(),
            float(self.step_count),
            terminated,
            truncated,
            {'raw_step': np.int32(self.step_count)},
        )

    def _obs(self):
        return np.array([self.step_count], dtype=np.float32)


def test_action_repeat_sums_rewards_and_returns_final_observation():
    env = ActionRepeatWrapper(CountingEnv(), repeat=3)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.5, -0.5], dtype=np.float32)
    )

    np.testing.assert_array_equal(obs, [3.0])
    assert reward == 1.0 + 2.0 + 3.0
    assert not terminated
    assert not truncated
    assert info['raw_step'] == 3
    assert info['repeat_count'] == 3
    assert len(env.unwrapped.actions) == 3
    for action in env.unwrapped.actions:
        np.testing.assert_array_equal(action, [0.5, -0.5])


def test_action_repeat_stops_on_termination():
    env = ActionRepeatWrapper(CountingEnv(terminated_at=2), repeat=5)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    np.testing.assert_array_equal(obs, [2.0])
    assert reward == 1.0 + 2.0
    assert terminated
    assert not truncated
    assert info['repeat_count'] == 2


def test_action_repeat_stops_on_truncation():
    env = ActionRepeatWrapper(CountingEnv(truncated_at=2), repeat=5)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    np.testing.assert_array_equal(obs, [2.0])
    assert reward == 1.0 + 2.0
    assert not terminated
    assert truncated
    assert info['repeat_count'] == 2


def test_action_repeat_reset_adds_default_repeat_count():
    env = ActionRepeatWrapper(CountingEnv(), repeat=3)

    _, info = env.reset()

    assert info['repeat_count'] == 0
    assert np.asarray(info['repeat_count']).dtype == np.int32


def test_action_repeat_one_behaves_like_single_step():
    env = ActionRepeatWrapper(CountingEnv(), repeat=1)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    np.testing.assert_array_equal(obs, [1.0])
    assert reward == 1.0
    assert not terminated
    assert not truncated
    assert info['repeat_count'] == 1
    assert env.unwrapped.step_count == 1


@pytest.mark.parametrize('repeat', [0, -1, 1.5])
def test_action_repeat_requires_positive_integer(repeat):
    with pytest.raises(ValueError):
        ActionRepeatWrapper(CountingEnv(), repeat=repeat)


def test_action_repeat_accumulates_inner_repeat_counts():
    env = ActionRepeatWrapper(
        ActionRepeatWrapper(CountingEnv(), repeat=2), repeat=3
    )
    env.reset()

    _, reward, _, _, info = env.step(env.action_space.sample())

    assert reward == sum(float(i) for i in range(1, 7))
    assert info['repeat_count'] == 6
    assert env.unwrapped.step_count == 6


def test_action_repeat_repeat_count_stacks_in_env_pool():
    pool = EnvPool(
        [
            lambda: ActionRepeatWrapper(CountingEnv(), repeat=3)
            for _ in range(2)
        ]
    )
    _, infos = pool.reset()

    np.testing.assert_array_equal(infos['repeat_count'], [[0], [0]])
    _, rewards, terminateds, truncateds, infos = pool.step(
        np.zeros((2, 2), dtype=np.float32)
    )

    np.testing.assert_array_equal(rewards, [6.0, 6.0])
    np.testing.assert_array_equal(terminateds, [False, False])
    np.testing.assert_array_equal(truncateds, [False, False])
    np.testing.assert_array_equal(infos['repeat_count'], [[3], [3]])
    pool.close()
