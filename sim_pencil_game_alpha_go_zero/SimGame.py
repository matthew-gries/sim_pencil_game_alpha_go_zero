import numpy as np
from typing import Tuple, Set

class SimGame:
    """
    Numpy implementation of the pencil game Sim
    """

    PLAYER1 = 1
    PLAYER2 = -1

    def __init__(self, adj: np.ndarray = None):
        """
        Construct a SimGame board.

        The adjacency list indices correspond to the following nodes on the graph

                    1
                -       -
            -               -
        0                       2
        |                       |
        |                       |
        5                       3
            -               -   
                -       -
                    4

        :param adj: the adjacency list representing the undirected graph. 0 represents the lack
            of an edge, 1 means player 1 has an edge, and -1 means player 2 has an edge. Defaults to None
        :type adj: np.ndarray, optional
        """
        self.adj = np.zeros((6, 6)) if adj is None else adj

    def get_legal_moves(self, player: int) -> Set[Tuple[int, int]]:
        """
        Get the legal moves this player can take. This is essentially any line that has not been
        drawn yet (outcome should be the same for any player)

        :param player: 1 if player 1, -1 if player 2
        :type player: int
        :return: the set of all moves, represented as tuples that denote the nodes to draw the edge between
        :rtype: Set[Tuple[int, int]]
        """
        moves = set()

        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                edge = self.adj[i][j]
                if edge == 0:
                    moves.add((i, j))
                    moves.add((j, i))

        return moves

    def draw_line(self, move: Tuple[int, int], player: int) -> bool:
        """
        Attempt to perform the given move with the given player

        :param move: tuple representing the nodes to create a line with
        :type move: Tuple[int, int]
        :param player: 1 for player 1, -1 for player 2
        :type player: int
        :return: True if the move could be made, False oetherwise
        :rtype: bool
        """

        if move[0] < 0 or move[0] > 5 or move[1] < 0 or move[1] > 5:
            return False

        # cant put a self-edge
        if move[0] == move[1]:
            return False

        edge1 = self.adj[move[0]][move[1]]
        edge2 = self.adj[move[1]][move[0]]

        if edge1 != 0 or edge2 != 0:
            return False

        self.adj[move[0]][move[1]] = player
        self.adj[move[1]][move[0]] = player

        return True

    def did_this_player_win(self, player: int) -> bool:
        """
        Check if the given player is currently the winner. This is done by seeing if
        the other player has a triangle; if they do, this player wins

        :param player: 1 for player 1, -1 for player 2
        :type player: int
        :return: True if the player won, False otherwise
        :rtype: bool
        """

        # mask matrix such that only the given player is present
        masked_board = np.where(self.adj==-player, 1, 0)
    
        triangle_count = np.trace(np.linalg.matrix_power(masked_board, 3)) / 6

        return triangle_count > 0

    def __str__(self) -> str:
        """
        Returns the graph as a string

        :return: graph as a string
        :rtype: str
        """

        p1_edges = set()
        p2_edges = set()

        for i in range(6):
            for j in range(6):
                edge = self.adj[i][j]
                if edge == 1:
                    p1_edges.add((i, j))
                elif edge == -1:
                    p2_edges.add((i, j))

        return f"P1 edges:\n\t{p1_edges}\nP2 edges:\n\t{p2_edges}"

    def string_rep(self) -> str:
        """
        Return a string representation used for hashing

        :return: the string representation used for hashing
        :rtype: str
        """
        return f"{self.adj.flatten().astype(str)}"