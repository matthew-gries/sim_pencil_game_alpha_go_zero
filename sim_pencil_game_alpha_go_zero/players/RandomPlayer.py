from sim_pencil_game_alpha_go_zero.players.AbstractPlayer import AbstractPlayer
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
import numpy as np

class RandomPlayer(AbstractPlayer):

    def __init__(self, game: SimGameState, player: int):
        super().__init__(game)
        self.player = player

    def play(self, board: np.ndarray) -> int:
        player = self.player
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, player)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a