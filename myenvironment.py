class MyEnvironment:
    def __init__(self, name, height, width, nchannelss):
        self._env = None
        if name == "my-Catch":
            from myenvs.catch import Catch
            self._env = Catch(width, height, nchannelss)
        else:
            raise ValueError("Unknown environment", name)

    def reset(self):
        return self._env.reset()

    def render(self):
        self._env.render()

    def step(self, action_index):
        return self._env.step(action_index)

    def get_num_actions(self):
        return self._env.get_num_actions()

    def monitor_start(self, path):
        print("monitor_start not implemented for custom environments")

    def monitor_close(self):
        print("monitor_close not implemented for custom environments")

if __name__ == "__main__":
    import random
    env = MyEnvironment("my-Catch", 84, 84, 4)
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = random.randint(0, env.get_num_actions()-1)
        state, reward, done, info = env.step(action)
        print(reward, done)
