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
            a = input()
            worker, move, build = a.split()
            a = self.game.atoi[(int(worker), move, build)]
            if a in self.game.legal_moves():
                break
            else:
                print('Invalid move')
        return a

class AlphaPlayer:
    def __init__(self, make_game, net, n_sims=100, c_puct=1, show_pred=False):
        self.game = make_game()
        self.net = net
        self.mcts = MCTS(make_game(), net, n_sims=n_sims, c_puct=c_puct)
        self.show_pred = show_pred

    def play(self, state):
        if self.show_pred:
            pi, v = self.net.predict(state)
            self.show_pi(pi)
            print('Value:', v)
        return np.argmax(self.mcts.get_action_prob(state, temp=0))
    
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