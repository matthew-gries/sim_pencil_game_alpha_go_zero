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
    line1, = ax1.plot(iteration, new_wins, color="blue")
    line2, = ax1.plot(iteration, prev_wins, color="red")
    line3, = ax1.plot(iteration, draws, color="green")
    ax1.legend([line1, line2, line3], ["New Model Wins", "Previous Model Wins", "Draws"])
    ax2.plot(iteration, accepts, color="tab:blue", marker="o", linestyle="None")
    ax2.set_xlabel("Iteration")
    ax1.set_ylabel("Count")
    ax2.set_ylabel("New Model Accepted")
    ax1.set_xticks(list(range(new_wins.shape[0])))
    ax2.set_yticks([0, 1])

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

    plt.show()



if __name__ == "__main__":
    # plot_win_draw_accept()
    plot_losses(2, 10)
