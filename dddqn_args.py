import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('experiment',
                    'breakout',
                    'Name of the current experiment')

flags.DEFINE_string('game',
                    'Breakout-v0',
                    'Name of the games. See `myenvironment.py` for custom games. '
                    'Full list of Atari games: https://gym.openai.com/envs#atari')

flags.DEFINE_integer('pre_training_steps',
                     1000000,
                     'The number of random steps before training')

flags.DEFINE_integer('num_training_episodes',
                     1000000,
                     'Number of training training.')

flags.DEFINE_integer('num_testing_episodes',
                     10,
                     'Number of episodes to run for evaluation.')

flags.DEFINE_integer('batch_size',
                     32,
                     'Batch size for update')

flags.DEFINE_string('trainer',
                    'RMSPropOptimizer',
                    'Training optimizer: [AdamOptimizer | RMSPropOptimizer '
                    '| AdadeltaOptimizer | AdagradOptimizer | GradientDescentOptimizer]')

flags.DEFINE_string('reward_adjustment_method',
                    'map',
                    'Method to adjust reward for training: [map | clip | none]')

flags.DEFINE_integer('width',
                     84,
                     'Scale frames to this width.')

flags.DEFINE_integer('height',
                     84,
                     'Scale frames to this height.')

flags.DEFINE_integer('num_channels',
                     4,
                     'The number of recent frames to the network.')

flags.DEFINE_integer('update_frequency',
                     4,
                     'Frequency to apply training step.')

flags.DEFINE_float('alpha',
                   0.0001,
                   'Learning rate.')

flags.DEFINE_float('gamma',
                   0.99,
                   'Reward discount rate.')

flags.DEFINE_float('tau',
                   0.001,
                   'Target network update rate.')

flags.DEFINE_integer('epsilon_annealing_steps',
                     100000,
                     'Number of steps to anneal epsilon.')

flags.DEFINE_float('start_epsilon',
                   1.0,
                   'Target network update rate.')

flags.DEFINE_float('final_epsilon',
                   0.1,
                   'Target network update rate.')

flags.DEFINE_integer('experience_buffer_size',
                     50000,
                     'Size of experience buffer')

flags.DEFINE_string('summary_dir',
                    '/tmp/summaries',
                    'Directory for storing tensorboard summaries.')

flags.DEFINE_string('checkpoint_dir',
                    '/tmp/checkpoints',
                    'Directory for storing model checkpoints.')

flags.DEFINE_integer('summary_interval',
                     20,
                     'Save training summary every n episode.')

flags.DEFINE_integer('checkpoint_interval',
                     1000,
                     'Save the model parameters every n episode.')

flags.DEFINE_boolean('show_training',
                     False,
                     'If true, have gym render the evironment during training.')

flags.DEFINE_boolean('load_model',
                     False,
                     'Continue training with a pre-trained model')

flags.DEFINE_boolean('testing',
                     False,
                     'If true, run for evaluation.')

flags.DEFINE_string('checkpoint_path',
                    'path/to/recent.ckpt',
                    'Path to recent checkpoint to use for evaluation or further training.')

flags.DEFINE_string('eval_dir',
                    '/tmp',
                    'Directory to store gym evaluation')

F = flags.FLAGS
