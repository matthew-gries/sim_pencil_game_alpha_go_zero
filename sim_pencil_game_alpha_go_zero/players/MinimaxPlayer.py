from sim_pencil_game_alpha_go_zero.players.AbstractPlayer import AbstractPlayer
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
import numpy as np

class MinimaxPlayer(AbstractPlayer):

    def __init__(self, game: SimGameState, player: int, max_depth: int):
        super(MinimaxPlayer, self).__init__(game)
        self.max_depth = max_depth
        self.player = player

    def heuristic(self, board: np.ndarray) -> float:
        """
        Heuristic for minimax, calculates the number of edges that are connected
        (more connected edges -> more opportunities to lose -> lesser value state)
        """

        masked_board = np.where(board==self.player, 1, 0)

        edges = set()

        for i in range(6):
            for j in range(6):
                edge = masked_board[i][j]
                if edge == 1:
                    if (j, i) not in edges:
                        edges.add((i, j))

        count = 0
        total = self.game.getActionSize() # same as number of edges total
        seen_nodes = set()
        
        for edge in edges:
            a, b = edge
            if a in seen_nodes:
                count += 1
            if b in seen_nodes:
                count += 1
            seen_nodes.add(a)
            seen_nodes.add(b)

        return self.player * (1 - (count / total))


    def play(self, board: np.ndarray) -> int:
        player = self.player
        valid_moves = self.game.getValidMoves(board, player)

        best_move = -1
        best_value = float('-inf') if (player == self.game.PLAYER1) else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        print("Running minimax...")
        for i in range(self.game.getActionSize()):
            if valid_moves[i] == 1:
                next_state, next_player = self.game.getNextState(board, player, i)
                value = self.minimax(next_state, next_player, self.max_depth, alpha, beta)
                if player == self.game.PLAYER1:
                    if value > best_value:
                        best_value = value
                        best_move = i
                else:
                    if value < best_value:
                        best_value = value
                        best_move = i
        return best_move


    # 1 is the maximizing player, -1 is the minimizing player
    def minimax(self, board: np.ndarray, player: int, current_depth: int, alpha: float, beta: float) -> float:

        if self.game.getGameEnded(board, player) != 0:
            return self.game.getGameEnded(board, player)
        elif current_depth == 0:
            return self.heuristic(board)
        elif player == self.game.PLAYER1:
            best_value = float('-inf')
            valid_moves = self.game.getValidMoves(board, player)
            for i in range(self.game.getActionSize()):
                if valid_moves[i] == 1:
                    next_board, next_player = self.game.getNextState(board, player, i)
                    value = self.minimax(next_board, next_player, current_depth-1, alpha, beta)
                    best_value = max(best_value, value)
                    if best_value >= beta:
                        break
                    alpha = max(alpha, best_value)
            return best_value
        else:
            best_value = float('inf')
            valid_moves = self.game.getValidMoves(board, player)
            for i in range(self.game.getActionSize()):
                if valid_moves[i] == 1:
                    next_board, next_player = self.game.getNextState(board, player, i)
                    value = self.minimax(next_board, next_player, current_depth-1, alpha, beta)
                    best_value = min(best_value, value)
                    if best_value <= alpha:
                        break
                    beta = min(beta, best_value)
            return best_value
