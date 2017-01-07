# Deep Reinforcement Learning with Tensorflow

## Setup

You need [anaconda][3], [tensorflow][4], [gym][5], and [tflearn][6] to run
this project.

Once you install anaconda, you might need the following commands helpful to create
and destroy your conda virtual environment.

```
$ export PATH=~/anaconda3/bin/:$PATH # Add it to your path.
$ conda -V # You can check conda version
$ conda update conda # You can update your conda
$ conda search "^python$" # See python versions
$ conda create -n <env-name> python=<version> anaconda # Create a virtual env
$ source activate <env-name> # Activate your virtual env
$ conda info -e # List your virtual envs
$ conda install -n <env-name> [package] # Install more packages
$ source deactivate # Deactivate current virtual env
$ conda remove -n <env-name> --all # Delete your environment <env-name>
```

Follow [tensorflow installation][4] on anaconda. If you want to use your GPU,
follow the Cuda installation and gpu-enabled tensorflow installation. If you are
installing gpu-enabled tensorflow, you might find [this][7] links helpful.
If you are building gpu-enabled tensorflow from scratch, follow [this][8] link.

TFLearn setup is as easy as running the following command.

```
pip install tflearn
```

Following commands install gym on your virtual environment.

```
$ apt-get install -y python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

$ pip install gym[atari]
```

I had to run the following commands to fix the problems that appeared after gym installation.

```
$ conda install -f numpy
$ conda install libgcc
```

## How to train

We have gym atari games and custom games. Custom games takes considerably
shorter time to train.

Here is a training command.
```
python dddqn.py train my-Catch \
--experiment=catch1 \ 
--num_random_steps=5000 \
--num_training_steps=2500 \
--num_validation_steps=1250 \
--epsilon_annealing_steps=50000 \
--experience_buffer_size=225000 \
--summary_dir=/tmp/summaries \
--checkpoint_dir=/tmp/checkpoints \
--target_update_frequency=5000 \
--tau=0.0 \
--alpha=0.00025
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
python plot.py /tmp/summaries/catch1/plot.csv --x_axis=epoch --y_axis=reward
python plot.py /tmp/summaries/catch1/plot.csv --x_axis=epoch --y_axis=maxq
python plot.py /tmp/summaries/catch1/plot.csv --x_axis=epoch --y_axis=epsilon
```

[1]: ./dddqn_args.py
[2]: https://gym.openai.com/envs#atari
[3]: https://www.continuum.io/downloads
[4]: https://www.tensorflow.org/get_started/os_setup#anaconda-installation
[5]: https://github.com/openai/gym
[6]: https://github.com/tflearn/tflearn
[7]: https://www.youtube.com/watch?v=io6Ajf5XkaM
[8]: https://alliseesolutions.wordpress.com/2016/09/08/install-gpu-tensorflow-from-sources-w-ubuntu-16-04-and-cuda-8-0-rc/
