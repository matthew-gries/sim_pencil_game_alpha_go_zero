from sim_pencil_game_alpha_go_zero.players.AbstractPlayer import AbstractPlayer
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
import numpy as np

class GreedyPlayer(AbstractPlayer):

    def __init__(self, game: SimGameState, player: int):
        super().__init__(game)
        self.player = player

    def play(self, board: np.ndarray) -> int:
        player = self.player
        valids = self.game.getValidMoves(board, player)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, player, a)
            score = self.game.getScore(nextBoard, player)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]