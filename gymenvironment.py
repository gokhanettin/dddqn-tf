from collections import deque
from copy import copy
import numpy as np
import cv2
import gym

class GymEnvironment:
    def __init__(self, name, height, width, nchannels):
        self._width = width
        self._height = height
        self._nchannels = nchannels
        self._env = gym.make(name)
        self._state = deque(maxlen=self._nchannels)

    def reset(self):
        self._state = deque(maxlen=self._nchannels)
        x = self._env.reset()
        x = self._get_preprocessed_frame(x)
        for _ in range(self._nchannels):
            self._state.append(x)
        state = copy(self._state)
        return state

    def render(self):
        self._env.render()

    def step(self, action):
        x, r, done, info = self._env.step(action)
        x = self._get_preprocessed_frame(x)
        self._state.append(x)
        state = copy(self._state)
        return state, r, done, info

    def get_num_actions(self):
        return self._env.action_space.n

    def monitor_start(self, path):
        self._env.monitor.start(path)

    def monitor_close(self):
        self._env.monitor.close()

    def _get_preprocessed_frame(self, x):
        return cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY),
                          (self._width, self._height))


if __name__ == "__main__":
    import random
    env = GymEnvironment("Breakout-v0", 84, 84, 4)
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = random.randint(0, env.get_num_actions()-1)
        state, reward, done, info = env.step(action)
        print(reward, done)
