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
    games = ["Breakout-v0", "SpaceInvaders-v0", "my-Catch", "my-Avoid"]
    for game in games:
        env = make_environment(game)
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = random.randrange(env.get_num_actions())
            state, reward, done, info = env.step(action)
            print(reward, done)
