from sim_pencil_game_alpha_go_zero.alphago_zero.NeuralNet import NeuralNet
from sim_pencil_game_alpha_go_zero.alphago_zero.utils import *
from sim_pencil_game_alpha_go_zero.SimGameState import SimGameState
from sim_pencil_game_alpha_go_zero.SimNNArch import SimNNArch

import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'checkpoint': './temp/',
})


class SimNN(NeuralNet):

    def __init__(self, game: SimGameState):
        self.game = game
        self.action_size = game.getActionSize()
        self.input_dim = (15, 3)
        self.nnet = SimNNArch(self.game, self.input_dim[0] * self.input_dim[1])
        self.device = 'cuda' if args.cuda else 'cpu'
        self.training_iter = 0

        if args.cuda:
            self.nnet.cuda()

    def make_one_shot(self, state: np.ndarray) -> torch.Tensor:
        """
        Convert the adjacency list representation of the graph into an `input_size` one shot encoding of each
        unique edge.

        Dimension one corresponds to the one-to-one mapping between each index and undirected edge, using the
        following encoding:
            Index 0 -> (0, 1),
            Index 1 -> (0, 2),
            ...
            Index 5 -> (1, 2),
            Index 6 -> (1, 3),
            ...
            Index 9 -> (2, 3),
            ...
            Index 13 -> (3, 5),
            Index 14 -> (4, 5)
        
        Dimension two corresponds to the spot to place the one, depending on who the edge belongs to. Index 0
        corresponds to player 1, index 1 corresponds to no edge, and index 2 corresponds to player 2.
        """

        one_shot = torch.zeros(self.input_dim).to(self.device, dtype=torch.float)
        for i in range(self.input_dim[0]):
            edge_tuple = self.game.ACTION_TO_TUPLE[i]
            edge = state[edge_tuple[0]][edge_tuple[1]]
            if edge == self.game.PLAYER1:
                one_shot[i][0] = 1
            elif edge == self.game.PLAYER2:
                one_shot[i][2] = 1
            else:
                one_shot[i][1] = 1
        
        return one_shot

    # Taken from othello example
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    # Taken from othello example
    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]  

    def saveLosses(self, pi_losses, v_losses):
        folder = args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        pi_loss_filename = os.path.join(folder, f"pi_losses{self.training_iter}.npy")
        v_loss_filename = os.path.join(folder, f"v_losses{self.training_iter}.npy")

        np.save(pi_loss_filename, np.array(pi_losses))
        np.save(v_loss_filename, np.array(v_losses))

        self.training_iter += 1

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        pi_losses_list = []
        v_losses_list = []

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards_list, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                boards = torch.zeros((len(boards_list), *self.input_dim)).to(self.device, dtype=torch.float)
                for i, board in enumerate(boards_list):
                    boards[i] = self.make_one_shot(board)
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Save average loss per epoch
            pi_losses_list.append(pi_losses.avg)
            v_losses_list.append(v_losses.avg)

        self.saveLosses(pi_losses_list, v_losses_list)

    def predict(self, board):
        # preparing input
        board_rep = self.make_one_shot(board).unsqueeze(0)
        if args.cuda:
            board_rep = board_rep.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board_rep)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
