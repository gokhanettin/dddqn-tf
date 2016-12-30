class MyEnvironment:
    def __init__(self, name, height, width, nchannels):
        self._env = None
        if name == "my-Catch":
            from myenvs.catch import Catch
            self._env = Catch(width, height, nchannels)
        elif name == "my-Avoid":
            from myenvs.avoid import Avoid
            self._env = Avoid(width, height, nchannels)
        else:
            raise ValueError("Unknown environment", name)

    def reset(self):
        return self._env.reset()

    def render(self):
        self._env.render()

    def step(self, action):
        return self._env.step(action)

    def get_num_actions(self):
        return self._env.get_num_actions()

    def monitor_start(self, path):
        self._env.monitor_start(path)

    def monitor_close(self):
        self._env.monitor_close()

if __name__ == "__main__":
    import random
    games = ["my-Catch", "my-Avoid"]
    for game in games:
        env = MyEnvironment(game , 84, 84, 4)
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = random.randrange(env.get_num_actions())
            state, reward, done, info = env.step(action)
            print(reward, done)
