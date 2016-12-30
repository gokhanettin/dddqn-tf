from collections import deque
import random
import numpy as np

class ExperienceBuffer:
    def __init__(self, maxlen):
        self._buff = deque(maxlen=maxlen)

    def append(self, experience):
        self._buff.append(experience)

    def extend(self, other):
        self._buff.extend(other._buff)

    def clear(self):
        self._buff.clear()

    def sample(self, batch_size):
        # [(s, a, r, s', done)
        #  (s, a, r, s', done)
        #  ...                )]
        batch = np.array(random.sample(self._buff, batch_size))
        return np.stack(batch[:, 0]), batch[:, 1], batch[:, 2], np.stack(batch[:, 3]), batch[:, 4]
