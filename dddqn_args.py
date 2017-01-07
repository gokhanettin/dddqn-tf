import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("game",
                              help="Environment id such as Breakout-v0, my-Catch etc.")
    train_parser.add_argument("--experiment", default="experiment",
                              help="Name of the current experiment.")
    train_parser.add_argument("--num_random_steps", type=int, default=50000,
                              help="Number of steps with random actions before training.")

    train_parser.add_argument("--num_noops_max", type=int, default=30,
                              help="Number of no ops steps when a new episode starts.")
    train_parser.add_argument("--num_epochs", type=int, default=200,
                              help="Number of epochs.")
    train_parser.add_argument("--num_training_steps", type=int, default=50000,
                              help="Number of training steps per epoch.")
    train_parser.add_argument("--num_validation_steps", type=int, default=25000,
                              help="Number of validation steps after each epoch.")
    train_parser.add_argument("--batch_size", type=int, default=32,
                              help="Minibatch size for network update.")
    train_parser.add_argument("--validation_epsilon", type=float, default=0.05,
                              help="e-greedy epsilon for validation.")
    train_parser.add_argument("--start_epsilon", type=float, default=1.0,
                              help="Initial epsilon during training.")
    train_parser.add_argument("--final_epsilon", type=float, default=0.1,
                              help="Final epsilon during training.")
    train_parser.add_argument("--epsilon_annealing_steps", type=int, default=1000000,
                              help="Number of steps to decay epsilon to its final value.")
    train_parser.add_argument("--experience_buffer_size", type=int, default=1000000,
                              help="How many experience items the buffer can hold.")
    train_parser.add_argument("--num_channels", type=int, default=4,
                              help="How many pre-processed frames in a state.")
    train_parser.add_argument("--online_update_frequency", type=int, default=4,
                              help="Frequency to update online params.")
    train_parser.add_argument("--target_update_frequency", type=int, default=10000,
                              help="Frequency to update target params.")
    train_parser.add_argument("--tau", type=float, default=0.0,
                              help="Target network update rate")
    train_parser.add_argument("--trainer",
                              choices=['adam', 'rmsprop', 'adadelta', 'adagrad', 'gradientdescent'],
                              default="adam",
                              help="Optimizer to train the network.")
    train_parser.add_argument("--reward_adjustment_method", choices=['clip', 'scale', 'none'],
                              default="clip",
                              help="Method to adjust reward for training.")
    train_parser.add_argument("--width", type=int, default=84,
                              help="Resized screen width.")
    train_parser.add_argument("--height", type=int, default=84,
                              help="Resized screen height.")
    train_parser.add_argument("--alpha", type=float, default=0.00025,
                              help="Learning rate.")
    train_parser.add_argument("--gamma", type=float, default=0.99,
                              help="Discount factor.")
    train_parser.add_argument("--checkpoint_path",
                              help="Continue training with a pre-trained model given by CHECKPOINT_PATH.")
    train_parser.add_argument("--summary_dir", default="/tmp/summaries",
                              help="Directory to save summaries.")
    train_parser.add_argument("--checkpoint_dir", default="/tmp/checkpoints",
                              help="Directory to save checkpoints.")
    train_parser.add_argument("--checkpoint_interval", metavar="N", type=int, default=50,
                              help="Save a checkpoint every N epoch")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("game",
                              help="Environment id such as Breakout-v0, my-Catch etc.")
    test_parser.add_argument("checkpoint_path",
                              help="Checkpoint path to load model")
    test_parser.add_argument("--test_epsilon", type=float, default=0.05,
                              help="e-greedy epsilon for test.")
    test_parser.add_argument("--eval_dir", default="/tmp/eval",
                              help="Directory to save evaluation results.")
    test_parser.add_argument("--num_testing_episodes", metavar="n", type=int, default=1,
                              help="Play the game n times.")
    test_parser.add_argument("--num_channels", type=int, default=4,
                              help="How many pre-processed frames in a state")
    test_parser.add_argument("--width", type=int, default=84,
                              help="Resized screen width.")
    test_parser.add_argument("--height", type=int, default=84,
                              help="Resized screen height.")
    test_parser.add_argument("--tau", type=float, default=0.0,
                              help="Target network update rate")
    test_parser.add_argument("--trainer",
                              choices=['adam', 'rmsprop', 'adadelta', 'adagrad', 'gradientdescent'],
                              default="adam",
                              help="Optimizer to train the network.")
    test_parser.add_argument("--alpha", type=float, default=0.00025,
                              help="Learning rate.")

    return parser.parse_args()

F = get_arguments()
