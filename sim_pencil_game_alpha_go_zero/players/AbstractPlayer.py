from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
import numpy as np

from abc import ABC, abstractmethod

class AbstractPlayer(ABC):

    def __init__(self, game: SimGameState):
        self.game = game

    @abstractmethod
    def play(self, board: np.ndarray) -> int:
        pass