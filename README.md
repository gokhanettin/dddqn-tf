# Deep Reinforcement Learning with Tensorflow

## How to train

We have gym atari games and custom games. Custom games takes considerably
shorter time to train.

Here is a training command.
```
python dddqn.py train \
my-Catch  \
--experiment=catch1 \
--pre_training_episodes=1000 \
--num_training_episodes=10000 \
--num_validation_episodes=5 \
--epsilon_annealing_episodes=5000 \
--experience_buffer_size=50000 \
--summary_interval=20 \
--checkpoint_interval=1000
```

See [dddqn_args.py][1] for all options.

Custom games are:

- my-Catch
- my-Avoid

Popular atari games are Breakout-v0, Pong-v0, SpaceInviders-v0, etc.
See [gym atari environments][2] for the full list of atari games.

You can use `tensorboard` to follow the training progress of the above command.

```
tensorboard --logdir=/tmp/summaries/catch1
```

Go to http://127.0.1.1:6006.

## How to test

```
python dddqn.py test my-Catch /tmp/checkpoints/catch1.ckpt-XXXX --eval_dir=/tmp/catch1
```

## How to plot

```
python plot.py /tmp/summaries/plot.py --x_axis=episode --y_axis=reward
python plot.py /tmp/summaries/plot.py --x_axis=episode --y_axis=maxq
python plot.py /tmp/summaries/plot.py --x_axis=episode --y_axis=epsilon
```



[1]: ./dddqn_args.py
[2]: https://gym.openai.com/envs#atari
