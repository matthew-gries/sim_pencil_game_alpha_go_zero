from sim_pencil_game_alpha_go_zero.alphago_zero.Arena import Arena
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
from sim_pencil_game_alpha_go_zero.players.NNPlayer import NNPlayer
from sim_pencil_game_alpha_go_zero.players.MinimaxPlayer import MinimaxPlayer
from sim_pencil_game_alpha_go_zero.players.RandomPlayer import RandomPlayer
from sim_pencil_game_alpha_go_zero.players.GreedyPlayer import GreedyPlayer
from sim_pencil_game_alpha_go_zero.alphago_zero.utils import dotdict

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_FOLDER = str(Path.home() / "Desktop" / "sim100iter100eps25sim10epoch")
MODEL_NAME = "best.pth.tar"

OLD_MODEL_FOLDER = str(Path.home() / "Desktop" / "sim100iter100eps25sim10epoch")
OLD_MODEL_NAME = "best.pth.tar"

def main():
    game = SimGameState()
    args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    model_player = NNPlayer(game, args, MODEL_FOLDER, MODEL_NAME)
    mp = MinimaxPlayer(game, max_depth=10, player=-1)
    gp = GreedyPlayer(game, player=-1)
    rp = RandomPlayer(game, player=-1)
    np = NNPlayer(game, args, OLD_MODEL_FOLDER, OLD_MODEL_NAME)
    mp_arena = Arena(model_player.play, mp.play, game, SimGameState.display)
    gp_arena = Arena(model_player.play, gp.play, game, SimGameState.display)
    rp_arena = Arena(model_player.play, rp.play, game, SimGameState.display)
    np_arena = Arena(model_player.play, np.play, game, SimGameState.display)

    mp_p1_wins, mp_p2_wins, _ = mp_arena.playGames(40, verbose=True)
    gp_p1_wins, gp_p2_wins, _ = gp_arena.playGames(40, verbose=True)
    rp_p1_wins, rp_p2_wins, _ = rp_arena.playGames(40, verbose=True)
    np_p1_wins, np_p2_wins, _ = np_arena.playGames(40, verbose=True)

    labels = ["Random Player", "Greedy Player", "Minimax Player", "Other Model"]
    p1_counts = [rp_p1_wins, gp_p1_wins, mp_p1_wins, np_p1_wins]
    p2_counts = [rp_p2_wins, gp_p2_wins, mp_p2_wins, np_p2_wins]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, p1_counts, width, label='Player 1')
    rects2 = ax.bar(x + width/2, p2_counts, width, label='Player 2')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Player 2 Player Type')
    ax.set_title('100 iterations, 1000 Episodes, 100 MCTS Sims vs Other Players')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
