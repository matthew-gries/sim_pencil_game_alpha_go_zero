import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np

# DATA_DIRECTORY = Path(__file__).parent / "alphago_zero" / "temp"
# DATA_DIRECTORY = Path.home() / "Desktop" / "trainsim100iter100eps10epoch25sim"
DATA_DIRECTORY = Path.home() / "Desktop" / "trainsim100iter1000eps20epoch100sim"


def plot_win_draw_accept():

    # iteration is the independent axis
    new_win_file = DATA_DIRECTORY / "new_wins.npy"
    prev_win_file = DATA_DIRECTORY / "prev_wins.npy"
    draws_file = DATA_DIRECTORY / "draws.npy"
    accepts_file = DATA_DIRECTORY / "accepts.npy"

    new_wins = np.load(str(new_win_file))
    prev_wins = np.load(str(prev_win_file))
    draws = np.load(str(draws_file))
    accepts = np.load(str(accepts_file))

    fig, axes = plt.subplots(3, 1, sharex=True)

    iteration = np.arange(0, new_wins.shape[0], step=1, dtype=int)

    total_games = new_wins[0] + prev_wins[0] + draws[0]
    ax1, ax2, ax3 = axes
    ax1.plot(iteration, new_wins / total_games, color="blue")
    ax2.plot(iteration, prev_wins / total_games, color="red")
    # line3, = ax1.plot(iteration, draws / total_games, color="green")
    ax3.plot(iteration, accepts, color="tab:blue", marker="o", linestyle="None")
    ax3.set_xlabel("Iteration")
    ax1.set_ylabel("New Model Win Percentage")
    ax2.set_ylabel("Previous Model Win Percentage")
    ax3.set_ylabel("New Model Accepted")
    ax3.set_yticks([0, 1])
    ax1.set_title("Win Rates per Iteration")

    plt.show()


def plot_losses(iterations, epochs):
    pi_losses = np.zeros((iterations, epochs))
    v_losses = np.zeros((iterations, epochs))

    for i in range(iterations):
        pi_loss = np.load(str(DATA_DIRECTORY / f"pi_losses{i}.npy"))
        pi_losses[i,:] = pi_loss
        v_loss = np.load(str(DATA_DIRECTORY / f"v_losses{i}.npy"))
        v_losses[i,:] = v_loss

    pi_losses_avg = np.average(pi_losses, axis=1)
    v_losses_avg = np.average(v_losses, axis=1)
    iterations = np.arange(0, pi_losses_avg.shape[0], step=1, dtype=int)
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax1, ax2 = ax
    ax1.plot(iterations, pi_losses_avg, color="blue")
    ax2.plot(iterations, v_losses_avg, color="red")
    ax2.set_xlabel("Iteration")
    ax1.set_ylabel("Average Loss over Epochs")
    ax2.set_ylabel("Average Loss over Epochs")
    ax1.set_title("Policy and Value Losses")

    plt.show()


def plot_min_max_avg_loss(iterations, epochs):
    pi_losses = np.zeros((iterations, epochs))
    v_losses = np.zeros((iterations, epochs))

    for i in range(iterations):
        pi_loss = np.load(str(DATA_DIRECTORY / f"pi_losses{i}.npy"))
        pi_losses[i,:] = pi_loss
        v_loss = np.load(str(DATA_DIRECTORY / f"v_losses{i}.npy"))
        v_losses[i,:] = v_loss

    pi_losses_avg = np.average(pi_losses, axis=1)
    v_losses_avg = np.average(v_losses, axis=1)
    pi_losses_min = np.min(pi_losses, axis=1)
    v_losses_min = np.min(v_losses, axis=1)
    pi_losses_max = np.max(pi_losses, axis=1)
    v_losses_max = np.max(v_losses, axis=1)
    iterations = np.arange(0, pi_losses_avg.shape[0], step=1, dtype=int)

    fig, axes = plt.subplots(2, 1, sharex=True)

    ax1, ax2 = axes
    line11, = ax1.plot(iterations, pi_losses_min, color="blue")
    line21, = ax1.plot(iterations, pi_losses_avg, color="red")
    line31, = ax1.plot(iterations, pi_losses_max, color="green")
    ax1.legend([line11, line21, line31], ["Min Loss", "Avg Loss", "Max Loss"])
    ax1.set_ylabel("Loss")
    ax1.set_title("Policy Loss")
    line12, = ax2.plot(iterations, v_losses_min, color="blue")
    line22, = ax2.plot(iterations, v_losses_avg, color="red")
    line32, = ax2.plot(iterations, v_losses_max, color="green")
    ax2.legend([line12, line22, line32], ["Min Loss", "Avg Loss", "Max Loss"])
    ax2.set_ylabel("Loss")
    ax2.set_title("Value Loss")

    plt.show()


if __name__ == "__main__":
    # plot_win_draw_accept()
    # plot_losses(100, 20)
    plot_min_max_avg_loss(100, 20)
