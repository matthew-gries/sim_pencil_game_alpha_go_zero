from sim_pencil_game_alpha_go_zero.players.AbstractPlayer import AbstractPlayer
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState

import numpy as np
from nuro_arm.robot.robot_arm import RobotArm

HOME_JPOS = [0.0041887902047863905, -0.6073745796940266, 1.26501464184549, 1.1770500475449757, -0.0041887902047863905]
JPOS_POSITIONS = {
    0: [-0.41469023027385266, -0.041887902047863905, 1.7928022076485752, 0.8712683625955693, -0.0041887902047863905],
    1: [0.0041887902047863905, -0.6408849013323178, 2.0943951023931953, 1.1016518238588207, -0.0041887902047863905],
    2: [0.4021238596594935, -0.041887902047863905, 1.7969909978533616, 0.8712683625955693, -0.0041887902047863905],
    3: [0.28483773392547457, 0.43563418129778464, 1.4158110892178, 0.5822418384653083, -0.0041887902047863905],
    4: [0.0041887902047863905, 0.632507320922745, 1.2733922222550627, 0.4063126498642799, -0.0041887902047863905],
    5: [-0.29321531433504733, 0.43563418129778464, 1.4116222990130136, 0.611563369898813, -0.0041887902047863905]
}

class RobotArmPlayer(AbstractPlayer):

    def __init__(self, game: SimGameState, driver: AbstractPlayer):
        super().__init__(game)
        self.driver = driver
        self.robot = RobotArm()
        self.robot.move_arm_jpos(HOME_JPOS)
        self.robot.close_gripper()

    def play(self, board: np.ndarray) -> int:
        action = self.driver.play(board)
        node_a, node_b = self.game.ACTION_TO_TUPLE[action]
        self.robot.move_arm_jpos(JPOS_POSITIONS[node_a])
        self.robot.move_arm_jpos(JPOS_POSITIONS[node_b])
        return action
