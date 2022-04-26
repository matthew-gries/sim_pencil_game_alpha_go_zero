from sim_pencil_game_alpha_go_zero.players.AbstractPlayer import AbstractPlayer
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
import numpy as np

def getScore(game: SimGameState, board: np.ndarray, player: int) -> float:
    """
    Same heuristic as minimax
    """
    masked_board = np.where(board==player, 1, 0)

    edges = set()

    for i in range(6):
        for j in range(6):
            edge = masked_board[i][j]
            if edge == 1:
                if (j, i) not in edges:
                    edges.add((i, j))

    count = 0
    total = game.getActionSize() # same as number of edges total
    seen_nodes = set()
    
    for edge in edges:
        a, b = edge
        if a in seen_nodes:
            count += 1
        if b in seen_nodes:
            count += 1
        seen_nodes.add(a)
        seen_nodes.add(b)

    return player * (1 - (count / total))

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
            nextBoard, nextPlayer = self.game.getNextState(board, player, a)
            score = getScore(self.game, nextBoard, nextPlayer)
            candidates += [(score, a)]
        reverse = True if (self.player == self.game.PLAYER1) else False
        candidates.sort(reverse=reverse)
        return candidates[0][1]