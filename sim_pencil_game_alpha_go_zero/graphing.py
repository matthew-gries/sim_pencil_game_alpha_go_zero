import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np

DATA_DIRECTORY = Path(__file__).parent / "alphago_zero" / "temp"


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

    fig, axes = plt.subplots(2, 1, sharex=True)

    iteration = np.arange(0, new_wins.shape[0], step=1, dtype=int)

    ax1, ax2 = axes
    line1, = ax1.plot(iteration, new_wins / new_wins.shape[0], color="blue")
    line2, = ax1.plot(iteration, prev_wins / prev_wins.shape[0], color="red")
    line3, = ax1.plot(iteration, draws / draws.shape[0], color="green")
    ax1.legend([line1, line2, line3], ["New Model Wins", "Previous Model Wins", "Draws"])
    ax2.plot(iteration, accepts, color="tab:blue", marker="o", linestyle="None")
    ax2.set_xlabel("Iteration")
    ax1.set_ylabel("Percentage")
    ax2.set_ylabel("New Model Accepted")
    ax1.set_xticks(list(range(new_wins.shape[0])))
    ax2.set_yticks([0, 1])
    ax1.set_title("Win and Draw Rates")

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
    fig, ax = plt.subplots()
    line1, = ax.plot(iterations, pi_losses_avg, color="blue")
    line2, = ax.plot(iterations, v_losses_avg, color="red")
    ax.legend([line1, line2], ["Policy Loss", "Value Loss"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Policy and Value Losses")

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

    fig, axes = plt.subplots(3, 1, sharex=True)

    ax1, ax2, ax3 = axes
    line11, = ax1.plot(iterations, pi_losses_min, color="blue")
    line21, = ax1.plot(iterations, v_losses_min, color="red")
    ax1.legend([line11, line21], ["Policy Loss", "Value Loss"])
    ax1.set_ylabel("Minimum Loss")
    line21, = ax2.plot(iterations, pi_losses_avg, color="blue")
    line22, = ax2.plot(iterations, v_losses_avg, color="red")
    ax2.legend([line21, line22], ["Policy Loss", "Value Loss"])
    ax2.set_ylabel("Average Loss")
    line31, = ax3.plot(iterations, pi_losses_max, color="blue")
    line32, = ax3.plot(iterations, v_losses_max, color="red")
    ax3.legend([line31, line32], ["Policy Loss", "Value Loss"])
    ax3.set_ylabel("Maximum Loss")
    ax3.set_xlabel("Iteration")
    ax1.set_title("Min/Avg/Max Loss Per Iteration")

    plt.show()


if __name__ == "__main__":
    # plot_win_draw_accept()
    # plot_losses(2, 10)
    plot_min_max_avg_loss(2, 10)
