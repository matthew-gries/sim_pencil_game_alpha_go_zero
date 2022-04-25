from sim_pencil_game_alpha_go_zero.players.AbstractPlayer import AbstractPlayer
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
from sim_pencil_game_alpha_go_zero.alphago_zero.utils import dotdict
from sim_pencil_game_alpha_go_zero.alphago_zero.MCTS import MCTS
from sim_pencil_game_alpha_go_zero.SimNN import SimNN

import numpy as np


class NNPlayer(AbstractPlayer):

    def __init__(self, game: SimGameState, args: dotdict, model_folder: str, model_name: str):
        super(NNPlayer, self).__init__(game)
        self.args = args
        self.nn = SimNN(game)
        self.nn.load_checkpoint(folder=model_folder, filename=model_name)
        self.mcts = MCTS(self.game, self.nn, self.args)

    def play(self, board: np.ndarray) -> int:
        return np.argmax(self.mcts.getActionProb(board, temp=0))
