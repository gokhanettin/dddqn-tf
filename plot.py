import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument("csv_file")
parser.add_argument("--x_axis", choices=['episode', 'step'],
                    default="episode",
                    help="X axis of the plot")
parser.add_argument("--y_axis", choices=['reward', 'maxq', 'epsilon'],
                    default="reward",
                    help="Y axis of the plot")

args = parser.parse_args()

dtype = [
    ("step", "int"),
    ("episode", "int"),
    ("validation_reward", "float"),
    ("validation_max_q", "float"),
    ("train_reward", "float"),
    ("train_max_q", "float"),
    ("epsilon", "float")
]

data = np.loadtxt(args.csv_file, skiprows=1, delimiter=",", dtype = dtype)

labels = {
    "step": "Step",
    "episode": "Episode",
    "reward": "Avrg. Reward",
    "maxq": "Avrg. Max Q",
    "epsilon": "Epsilon"
}

fig, ax = plt.subplots()
if args.y_axis == "reward":
    ax.plot(data[args.x_axis], data["train_reward"], label='Train', linestyle='--')
    ax.plot(data[args.x_axis], data["validation_reward"], label='Validation')
    ax.legend()
elif args.y_axis == "maxq":
    ax.plot(data[args.x_axis], data["train_max_q"], label='Train', linestyle='--')
    ax.plot(data[args.x_axis], data["validation_max_q"], label='Validation')
    ax.legend()
else:
    ax.plot(data[args.x_axis], data["epsilon"])
ax.set_xlabel(labels[args.x_axis])
ax.set_ylabel(labels[args.y_axis])
plt.show()

