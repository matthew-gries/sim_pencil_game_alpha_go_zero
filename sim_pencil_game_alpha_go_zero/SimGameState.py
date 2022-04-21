from sim_pencil_game_alpha_go_zero.alphago_zero.Game import Game
from sim_pencil_game_alpha_go_zero.SimGame import SimGame

import numpy as np
from typing import Tuple
import logging


class SimGameState(Game):

    PLAYER1 = 1
    PLAYER2 = -1

    ACTION_TO_TUPLE = {
        0: (0, 1),
        1: (0, 2),
        2: (0, 3),
        3: (0, 4),
        4: (0, 5),
        5: (1, 2),
        6: (1, 3),
        7: (1, 4),
        8: (1, 5),
        9: (2, 3),
        10: (2, 4),
        11: (2, 5),
        12: (3, 4),
        13: (3, 5),
        14: (4, 5)
    }

    def getInitBoard(self):
        sg = SimGame()
        return sg.adj

    def getBoardSize(self):
        return (6, 6)

    def getActionSize(self):
        return 15

    def numeric_action_to_tuple(self, action: int) -> Tuple[int, int]:
        return self.ACTION_TO_TUPLE[action]

    def getNextState(self, board: np, player, action):
        if 0 <= action <= 14:
            logging.error(f"Invalid action {action} given!")
            return board, player

        new_board = board.copy()
        sg = SimGame(new_board)

        move = self.numeric_action_to_tuple(action)

        if not sg.draw_line(move, player):
            logging.error(f"Move {move} could not be taken by player {player}!")
            return board, player

        return sg.adj, -player

    def getValidMoves(self, board, player):
        # player doesn't actually matter for valid moves
        valids = [0]*self.getActionSize()
        sg = SimGame(board)
        tuple_to_action = dict(zip(self.ACTION_TO_TUPLE.values(), self.ACTION_TO_TUPLE.keys()))
        legal_moves = sg.get_legal_moves(player)
        for legal_move in legal_moves:
            action_number = tuple_to_action[legal_move]
            valids[action_number] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        sg = SimGame(board)

        if sg.did_this_player_win(player):
            return 1
        elif sg.did_this_player_win(-player):
            return -1
        # player doesn't matter
        elif sg.get_legal_moves(player):
            return 0
        else:
            # draw condition, not sure if we can actually reach this
            return 1e-4

    def getCanonicalForm(self, board, player):
        return board*player

    def getSymmetries(self, board, pi):
        # Don't worry about this for now
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tostring()
