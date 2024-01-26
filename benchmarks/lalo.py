import gym
import numpy as np
from typing import Tuple, Dict, Any
import math

class LaloeEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,))
        self.observation_space = gym.spaces.Box(
            low=np.array([-3.8, -3.95, -3.50, -2.60, -4.00, -4.90, -4.55]), high=np.array([6.20, 6.05, 6.50, 7.40, 6.00, 5.10, 5.45])
        )

        self.init_space = gym.spaces.Box(
            low=np.array([1.15, 1.00, 1.45, 2.35, 0.95, 0.05, 0.40]), high=np.array([1.25, 1.10, 1.55, 2.45, 1.05, 0.15, 0.50])
        )

        self.rng = np.random.default_rng()

        self._max_episode_steps = 100

        self.x1_threshold_lower = -3.3
        self.x1_threshold_upper = 1.25
        self.x2_threshold_lower = -3.45
        self.x2_threshold_upper = 1.1
        self.x3_threshold_lower = -3
        self.x3_threshold_upper = 1.55
        self.x4_threshold_lower = -2.1
        self.x4_threshold_upper = 1.05
        self.x5_threshold_lower = -3.5
        self.x5_threshold_upper = 1.05
        self.x6_threshold_lower = -4.4
        self.x6_threshold_upper = 1.05
        self.x7_threshold_lower = -4.05
        self.x7_threshold_upper = 0.5
        self.polys = [
            np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1 * self.x1_threshold_upper],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.x1_threshold_lower],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1 * self.x2_threshold_upper],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.x2_threshold_lower],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1 * self.x3_threshold_upper],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, self.x3_threshold_lower],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1 * self.x4_threshold_upper],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, self.x4_threshold_lower],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1 * self.x5_threshold_upper],
                [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, self.x5_threshold_lower],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1 * self.x6_threshold_upper],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, self.x6_threshold_lower],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1 * self.x7_threshold_upper],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, self.x7_threshold_lower],
            ])
        ]

        self.safe_polys = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01 - self.x1_threshold_lower]),
            np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01 + self.x1_threshold_upper]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01 - self.x2_threshold_lower]),
            np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01 + self.x2_threshold_upper]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.01 - self.x3_threshold_lower]),
            np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.01 + self.x3_threshold_upper]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.01 - self.x4_threshold_lower]),
            np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.01 + self.x4_threshold_upper]),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.01 - self.x5_threshold_lower]),
            np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.01 + self.x5_threshold_upper]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.01 - self.x6_threshold_lower]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.01 + self.x6_threshold_upper]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.01 - self.x7_threshold_lower]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.01 + self.x7_threshold_upper]),
        ]

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def f(self, state, u):
        x1, x2, x3, x4, x5, x6, x7 = state
        
        x1_dot = 1.4 * x3 - 0.9 * x1
        x2_dot = 2.5 * x5 - 1.5 * x2 + u
        x3_dot = 0.6 * x7 - 0.8 * x2 * x3
        x4_dot = 2 - 1.3 * x3 * x4
        x5_dot = 0.7 * x1 - x4 * x5
        x6_dot = 0.3 * x1 - 3.1 * x6
        x7_dot = 1.8 * x6 - 1.5 * x2 * x7
        return np.array([x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot])

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        dt = 0.1
        fxu = self.f(self.state, action[0])

        self.state = self.state + dt * fxu


        reward = -1 * np.linalg.norm(fxu).item()
        if self.unsafe(self.state):
            reward -= 10

        self.steps += 1
        done = self.steps >= self._max_episode_steps
        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return False
        # return state[0] >= 3.0 and state[1] >= 3.0

    def true_reward(self, state, corner):
        import pdb
        pdb.set_trace()
        fxu = self.f(state, corner[0])
        reward = -1 * np.linalg.norm(fxu).item()
        if self.unsafe(state):
            reward -= 10
        return reward

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self, state) -> bool:
            lower_bound = np.array([-3.30, -3.45, -3.00, -2.10, -3.50, -4.40, -4.05])
            upper_bound = np.array([1.25, 1.10, 1.55, 2.45, 1.05, 1.05, 0.50])
            return np.all(lower_bound <= state).item() and np.all(state <= upper_bound).item()
