import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SantoriniAI.alphazero.mcts import MCTS
sns.set()

class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, state):
        return np.random.choice(self.game.legal_moves())

class HumanPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, state):
        while True:
            try:
                a = input('worker move build')
                worker, move, build = a.split()
                a = self.game.atoi[(abs(int(worker)), move, build)]
                if a in self.game.legal_moves():
                    break
                else:
                    print('Illegal move')
            except:
                print('Invalid format')
        return a

class AlphaPlayer:
    def __init__(self, make_game, net, n_sims=100, c_puct=1, show_pred=False, prepare_state=False):
        self.game = make_game()
        self.net = net
        self.mcts = MCTS(make_game(), net, n_sims=n_sims, c_puct=c_puct)
        self.show_pred = show_pred
        self.prepare_state = prepare_state

    def play(self, state):
        if self.prepare_state:
            state = self.prepare(state)
        if self.show_pred:
            pi, v = self.net.predict(state)
            self.show_pi(pi)
            print('Value:', v)
        return np.argmax(self.mcts.get_action_prob(state, temp=0))
    
    def prepare(self, state):
        board = state[0].astype(np.int8)
        wboard = state[1].astype(np.int8)
        parts = state[2].diagonal()
        workers = np.concatenate([[np.where(state[1] == i)] for i in [-1, -2, 1, 2]])[:, :, 0]
        current_player = -1
        return (board, wboard, parts, workers, current_player)
    
    def show_pi(self, pi):
        pi = np.reshape(np.round(pi, decimals=3), (2, 8, 8))

        plt.figure(figsize=(6, 6))
        sns.heatmap(pi[0], annot=True, square=True)
        plt.xticks(np.arange(8) + 0.5, self.game.moves, rotation=0)
        plt.yticks(np.arange(8) + 0.5, self.game.builds, rotation=0)
        plt.xlim(0, 8)
        plt.ylim(8, 0)
        plt.show()

        plt.figure(figsize=(6, 6))
        sns.heatmap(pi[1], annot=True, square=True)
        plt.xticks(np.arange(8) + 0.5, self.game.moves, rotation=0)
        plt.yticks(np.arange(8) + 0.5, self.game.builds, rotation=0)
        plt.xlim(0, 8)
        plt.ylim(8, 0)
        plt.show()