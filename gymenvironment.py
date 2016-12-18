from collections import deque
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import gym

class GymEnvironment:
    def __init__(self, name, height, width, nchannels):
        self._width = width
        self._height = height
        self._nchannels = nchannels
        self._env = gym.make(name)
        self._state = deque(maxlen=self._nchannels-1)
        if (self._env.spec.id == "Pong-v0" or self._env.spec.id == "Breakout-v0"):
            print("Doing workaround for pong or breakout")
            # Gym returns 6 possible actions for breakout and pong.
            # Only three are used, the rest are no-ops. This just lets us
            # pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
            self._gym_actions = [1,2,3]
        else:
            self._gym_actions = range(self._env.get_num_actions())

    def reset(self):
        self._state = deque(maxlen=self._nchannels-1)
        x = self._env.reset()
        x = self._get_preprocessed_frame(x)
        s = np.stack(([x] * self._nchannels), axis = 0)
        for _ in range(self._nchannels-1):
            self._state.append(x)
        return s

    def render(self):
        self._env.render()

    def step(self, action_index):
        x, r, done, info = self._env.step(self._gym_actions[action_index])
        x = self._get_preprocessed_frame(x)
        previous_frames = np.array(self._state)
        s = np.empty((self._nchannels, self._height, self._width))
        s[:self._nchannels-1] = previous_frames
        s[self._nchannels-1] = x
        self._state.append(x)
        return s, r, done, info

    def get_num_actions(self):
        return len(self._gym_actions)

    def monitor_start(self, path):
        self._env.monitor.start(path)

    def monitor_close(self):
        self._env.monitor.close()

    def _get_preprocessed_frame(self, x):
        return resize(rgb2gray(x), (self._width, self._height))


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
