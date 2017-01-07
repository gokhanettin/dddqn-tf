from __future__ import print_function
import itertools
from copy import copy
from collections import deque
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.animation as animation


GRIDSIZE = 5

_RED = 0
_GREEN = 1
_BLUE = 2

_UP = 0
_DOWN = 1
_LEFT = 2
_RIGHT = 3

class _Object:
    def __init__(self, coordinates, size, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.channel = channel
        self.reward = reward
        self.name = name

class Catch:
    def __init__(self, height, width, nchannels):
        self._size_x = GRIDSIZE
        self._size_y = GRIDSIZE
        self._objects = []
        self._step_count = 0
        self._max_step = GRIDSIZE * GRIDSIZE * 2
        self._width = width
        self._height = height
        self._nchannels = nchannels
        self._writer = None
        self.reset()

    def reset(self):
        self._objects = []
        agent = _Object(self._random_position(), 1, _BLUE, None, 'agent')
        self._objects.append(agent)
        food1 = _Object(self._random_position(), 1, _GREEN, 1, 'food')
        self._objects.append(food1)

        self._rendered = False
        self._state = deque(maxlen=self._nchannels)
        x = self._get_preprocessed_frame()
        for _ in range(self._nchannels):
            self._state.append(x)
        state = copy(self._state)
        return state

    def render(self):
        if not self._rendered:
            self._rendered = True
            self._imshow = plt.imshow(self._frame, interpolation="nearest")
            plt.title("Catch Game")
        else:
            self._imshow.set_data(self._frame)
        plt.pause(0.3)
        if self._writer:
            self._writer.grab_frame()

    def step(self, action):
        self._move(action)
        r, done = self._check_goal()
        x = self._get_preprocessed_frame()
        self._state.append(x)
        state = copy(self._state)
        return state, r, done, None

    def get_num_actions(self):
        # up, down, right, left
        return 4

    def monitor_start(self, path):
        import os
        os.makedirs(path)
        video_file = os.path.join(path, "catch_video.mp4")
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Catch Game', artist='DRL',
                        comment='Simple Grid Game for DRL')
        self._writer = FFMpegWriter(fps=2, metadata=metadata)

        fig = plt.figure()
        self._writer.setup(fig, video_file, 100)

    def monitor_close(self):
        self._writer.finish()

    def _mk_frame(self):
        a = np.ones([self._size_y+2, self._size_x+2, 3], dtype=np.uint8)
        a[1:-1, 1:-1, :] = 0
        for obj in self._objects:
            a[obj.y+1:obj.y+obj.size+1, obj.x+1:obj.x+obj.size+1, obj.channel] = 255
        r = scipy.misc.imresize(a[:,:,0],[84,84,1],interp='nearest')
        g = scipy.misc.imresize(a[:,:,1],[84,84,1],interp='nearest')
        b = scipy.misc.imresize(a[:,:,2],[84,84,1],interp='nearest')
        self._frame = np.stack([r, g, b],axis=2)
        return self._frame

    def _random_position(self):
        iterables = [ range(self._size_x), range(self._size_y)]
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
        if direction == _DOWN and agent.y <= self._size_y-2:
            agent.y += 1
        if direction == _LEFT and agent.x >= 1:
            agent.x -= 1
        if direction == _RIGHT and agent.x <= self._size_x-2:
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
                    self._objects.append(_Object(self._random_position(), 1, _GREEN, 1, 'food'))
                return other.reward, done
        return 0, done

    def _get_preprocessed_frame(self):
        self._mk_frame()
        # http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
        gray = np.dot(self._frame[..., :3], [0.299, 0.587, 0.114])/255.0
        return imresize(gray, (self._width, self._height), interp='nearest')



if __name__ == "__main__":
    import random
    env = Catch(84, 84, 4)
    state = env.reset()
    done = False
    env.monitor_start("/tmp/catch/eval")
    while not done:
        env.render()
        action = random.randrange(env.get_num_actions())
        state, reward, done, info = env.step(action)
        print(reward, done)
    env.monitor_close()
