from sim_pencil_game_alpha_go_zero.players.AbstractPlayer import AbstractPlayer
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
import numpy as np

class HumanPlayer(AbstractPlayer):

    def __init__(self, game: SimGameState, player: int):
        super().__init__(game)
        self.player = player

    def play(self, board: np.ndarray) -> int:
        player = self.player
        valids = self.game.getValidMoves(board, player)
        print("Moves:")
        for i in range(self.game.getActionSize()):
            if valids[i] == 1:
                move = self.game.ACTION_TO_TUPLE[i]
                print(f"\t{i} -> {move}")

        print("Select an action")
        idx = -1
        while True:
            try:
                idx = int(input())
            except Exception as e:
                print(f"Error: {e}. Please try again")
                continue
            if valids[idx] != 1:
                print("Invalid move requested. Please try again")
                continue
            break

        return idx
        
        
        
