from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
import numpy as np
from abc import ABC, abstractmethod

class AbstractPlayer(ABC):

    def __init__(self, game: SimGameState):
        self.game = game

    @abstractmethod
    def play(self, board: np.ndarray) -> int:
        pass

class RandomPlayer(AbstractPlayer):

    def play(self, board: np.ndarray) -> int:
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a