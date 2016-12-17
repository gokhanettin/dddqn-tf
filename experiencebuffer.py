from collections import deque
import random
import numpy as np

class ExperienceBuffer:
    def __init__(self, maxlen):
        self.buff = deque(maxlen=maxlen)

    def add(self, experience):
        self.buff.extend(experience)

    def sample(self, batch_size):
        # [[s', a, r, s, done']
        #  [s', a, r, s, done']
        #  ...                ]]
        experience = np.array(random.sample(self.buff, batch_size))
        return experience
