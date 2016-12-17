def make_environment(name, height=84, width=84, nchannels=4):
    env = None
    if name.split('-')[0] == 'my':
        from myenvironment import MyEnvironment
        env = MyEnvironment(name, height, width, nchannels)
    else:
        from gymenvironment import GymEnvironment
        env = GymEnvironment(name, height, width, nchannels)
    return env

if __name__ == "__main__":
    import random
    env = make_environment('Breakout-v0')
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = random.randint(0, env.get_num_actions()-1)
        state, reward, done, info = env.step(action)
        print(reward, done)

    env = make_environment('my-Catch')
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = random.randint(0, env.get_num_actions()-1)
        state, reward, done, info = env.step(action)
        print(reward, done)

