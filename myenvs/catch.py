from __future__ import print_function
import itertools
from collections import deque
import numpy as np
import scipy.misc
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


GRIDSIZE = 5

_RED = 0
_GREEN = 1
_BLUE = 2

_UP = 0
_DOWN = 1
_LEFT = 2
_RIGHT = 3

class _Object:
    def __init__(self, coordinates, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.channel = channel
        self.reward = reward
        self.name = name

class Catch:
    def __init__(self, height, width, nchannels):
        self._sizeX = GRIDSIZE
        self._sizeY = GRIDSIZE
        self._objects = []
        self._step_count = 0
        self._max_step = GRIDSIZE * GRIDSIZE * 2
        self._width = width
        self._height = height
        self._nchannels = nchannels
        self.reset()
        plt.ion() # plot interactive mode on

    def reset(self):
        self._objects = []
        agent = _Object(self._random_position(), _BLUE, None, 'agent')
        self._objects.append(agent)
        food1 = _Object(self._random_position(), _GREEN, 1, 'food')
        self._objects.append(food1)

        self._state = deque(maxlen=self._nchannels-1)
        x = self._mk_image()
        x = self._get_preprocessed_frame(x)
        s = np.stack(([x] * self._nchannels), axis = 0)
        for _ in range(self._nchannels-1):
            self._state.append(x)
        return s

    def render(self):
        plt.imshow(self._image, interpolation="nearest")
        plt.pause(0.05)

    def step(self, action):
        self._move(action)
        r, done = self._check_goal()
        x = self._mk_image()
        x = self._get_preprocessed_frame(x)
        previous_frames = np.array(self._state)
        s = np.empty((self._nchannels, self._height, self._width))
        s[:self._nchannels-1] = previous_frames
        s[self._nchannels-1] = x
        self._state.append(x)
        return s, r, done, None

    def get_num_actions(self):
        # up, down, right, left
        return 4

    def _mk_image(self):
        a = np.ones([self._sizeY+2, self._sizeX+2, 3])
        a[1:-1, 1:-1, :] = 0
        for obj in self._objects:
            a[obj.y+1:obj.y+2, obj.x+1:obj.x+2, obj.channel] = 1
        r = scipy.misc.imresize(a[:,:,0],[84,84,1],interp='nearest')
        g = scipy.misc.imresize(a[:,:,1],[84,84,1],interp='nearest')
        b = scipy.misc.imresize(a[:,:,2],[84,84,1],interp='nearest')
        self._image = np.stack([r, g, b],axis=2)
        return self._image

    def _random_position(self):
        iterables = [ range(self._sizeX), range(self._sizeY)]
        points = []
        for point in itertools.product(*iterables):
            points.append(point)
        current_positions = []
        for obj in self._objects:
            if (obj.x, obj.y) not in current_positions:
                current_positions.append((obj.x, obj.y))
        for pos in current_positions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def _move(self, direction):
        agent = self._objects[0]
        agentX = agent.x
        agentY = agent.y
        if direction == _UP and agent.y >= 1:
            agent.y -= 1
        if direction == _DOWN and agent.y <= self._sizeY-2:
            agent.y += 1
        if direction == _LEFT and agent.x >= 1:
            agent.x -= 1
        if direction == _RIGHT and agent.x <= self._sizeX-2:
            agent.x += 1
        self._objects[0] = agent


    def _check_goal(self):
        others = []
        for obj in self._objects:
            if obj.name == 'agent':
                agent = obj
            else:
                others.append(obj)
        self._step_count += 1
        done = False
        if self._step_count  >= self._max_step:
            done = True
            self._step_count = 0
        for other in others:
            if agent.x == other.x and agent.y == other.y:
                self._objects.remove(other)
                if other.reward == 1:
                    self._objects.append(_Object(self._random_position(), _GREEN, 1, 'goal'))
                return other.reward, done
        return 0, done

    def _get_preprocessed_frame(self, x):
        return resize(rgb2gray(x), (self._width, self._height))

if __name__ == "__main__":
    import random
    env = Catch(84, 84, 4)
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = random.randint(0, env.get_num_actions()-1)
        state, reward, done, info = env.step(action)
        print(reward, done)
