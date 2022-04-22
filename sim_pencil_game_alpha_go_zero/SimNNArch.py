from typing import Tuple
import torch
import torch.nn as nn

from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState

class SimNNArch(nn.Module):

    def __init__(self, game: SimGameState, input_size: int):
        super(SimNNArch, self).__init__()
        self.input_size = input_size
        self.game = game

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.policy_fc = nn.Linear(256, self.game.getActionSize())
        self.value_fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x should be the one-hot encoding of each unique edge, size 15 x 3,
        where dimension 1 is the index corresponding to the edge in the graph and
        dimension 2 has a 1 at index 0 if player 1 has an edge, index 1 = 1 if
        the edge has not been drawn yet, index 2 = 1if player 2 has an edge

        Returns the policy vector and the value of the state
        """

        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        pi = torch.log_softmax(self.policy_fc(x), dim=0)
        v = torch.tanh(self.value_fc(x))

        return pi, v