from sim_pencil_game_alpha_go_zero.alphago_zero.Arena import Arena
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
from sim_pencil_game_alpha_go_zero.players.HumanPlayer import HumanPlayer
from sim_pencil_game_alpha_go_zero.players.NNPlayer import NNPlayer
from sim_pencil_game_alpha_go_zero.players.MinimaxPlayer import MinimaxPlayer
from sim_pencil_game_alpha_go_zero.players.RandomPlayer import RandomPlayer
from sim_pencil_game_alpha_go_zero.alphago_zero.utils import dotdict

from pathlib import Path

MODEL_FOLDER = str(Path.home() / "Desktop" / "sim100iter100eps25sim10epoch")

MODEL_NAME = "best.pth.tar"


def main():
    game = SimGameState()
    args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    # p1 = NNPlayer(game, args, MODEL_FOLDER, MODEL_NAME)
    p1 = MinimaxPlayer(game, max_depth=5, player=1)
    # p2 = HumanPlayer(game, player=-1)
    p2 = RandomPlayer(game, player=-1)
    arena = Arena(p1.play, p2.play, game, SimGameState.display)

    p1_wins, p2_wins, draws = arena.playGames(10, verbose=True)

    print(f"PLAYER 1 WINS: {p1_wins}\nPLAYER 2 WINS: {p2_wins}\nDRAWS: {draws}")


if __name__ == "__main__":
    main()