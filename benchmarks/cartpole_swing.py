import gym
import numpy as np
from typing import Tuple, Dict, Any
import math

class CartPoleSwingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 1.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 1.5
        self.x_threshold = 0.9

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(-10, 10, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.init_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(4,))

        self._max_episode_steps = 100

        self.polys = [
            np.array([[1.0, 0.0, 0.0, 0.0, self.x_threshold]]),
            np.array([[-1.0, 0.0, 0.0, 0.0, self.x_threshold]]),
            np.array([[0.0, 0.0, 1.0, 0.0, self.theta_threshold_radians]]),
            np.array([[0.0, 0.0, -1.0, 0.0, self.theta_threshold_radians]])
        ]

        self.safe_polys = [
            np.array([
                [-1.0, 0.0, 0.0, 0.0, -1*self.x_threshold + 0.01],
                [1.0, 0.0, 0.0, 0.0, -1*self.x_threshold + 0.01],
                [0.0, 0.0, -1.0, 0.0, -1* self.theta_threshold_radians + 0.01],
                [0.0, 0.0, 1.0, 0.0, -1* self.theta_threshold_radians + 0.01],
            ])
        ]

    def reset(self) -> np.ndarray:
        self.state = self.init_space.sample()
        self.steps = 0
        return self.state

    def step(self, action: np.ndarray) -> \
            Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag * action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        # if not self.unsafe():
        #     reward = 1.0
        # else:
        #     reward = 0.0
        reward = (self.state[2])**2
        if self.unsafe(self.state):
            reward -= 30

        self.steps += 1
        done = self.steps >= self._max_episode_steps
        return self.state, reward, done, {}

    def predict_done(self, state: np.ndarray) -> bool:
        return False
        # return state[0] >= 3.0 and state[1] >= 3.0

    def true_reward(self, state, corner):
        reward = (self.state[2])**2
        if self.unsafe(state):
            reward = -30
        return reward

    def seed(self, seed: int):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.init_space.seed(seed)

    def unsafe(self, state: np.ndarray) -> bool:
        return state[0] < -1 * self.x_threshold or state[0] > self.x_threshold or \
            state[2] < -1 * self.theta_threshold_radians or state[2] > self.theta_threshold_radians
