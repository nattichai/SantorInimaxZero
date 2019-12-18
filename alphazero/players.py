import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .mcts import MCTS
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
                a = input('Input your move (worker move build): ')
                worker, move, build = a.split()
                a = self.game.atoi[(abs(int(worker)), move, build)]
                if a in self.game.legal_moves():
                    break
                else:
                    print('Illegal move')
            except KeyboardInterrupt:
                raise
            except:
                print('Invalid format')
        return a

class AlphaPlayer:
    def __init__(self, make_game, net, n_sims=100, c_puct=1, show_pred=False, prepare_state=False):
        self.game = make_game()
        self.net = net
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.mcts = MCTS(make_game(), net, n_sims=n_sims, c_puct=c_puct)
        self.show_pred = show_pred
        self.prepare_state = prepare_state
        self.action_prob = None

    def play(self, state):
        if self.prepare_state:
            state = prepare(state)
        if self.show_pred:
            pi, v = self.net.predict(state)
            print('AlphaZero value:', v)
            show_pi(pi)
        self.action_prob = self.mcts.get_action_prob(state, temp=1)
        return np.argmax(self.action_prob)
        
class MinimaxPlayer:
    def __init__(self, make_game, depth=4, show_pred=False, prepare_state=False):
        self.game = make_game()
        self.depth = depth
        self.show_pred = show_pred
        self.prepare_state = prepare_state
        self.memory = {}
        self.values_cache = None

    def play(self, state):
        if self.prepare_state:
            state = prepare(state)
        self.game.set_state(state)
        self.values_cache = self.negamax(state, self.depth, legal_moves=self.game.legal_moves())
        if self.show_pred:
            print('Minimax value:', self.values_cache.max())
            show_pi(self.values_cache)
        return np.random.choice(np.argwhere(self.values_cache == self.values_cache.max()).ravel())

    def negamax(self, state, depth, alpha=-1, beta=1, legal_moves=None):
        s = self.game.tostring(state)
        if depth == self.depth:
            values = np.full(128, -2)
        elif s in self.memory and depth <= self.memory[s][0]:
            return self.memory[s][1]
        elif depth == 0:
            return 0
        
        for action in legal_moves:
            self.game.set_state(state)
            next_state, reward, done, next_legals = self.game.step(action)
            
            if done:
                v = reward
            elif self.game._reward_cache != 0:
                v = -self.game._reward_cache
            else:
                v = -self.negamax(next_state, depth - 1, -beta, -alpha, next_legals)
                            
            alpha = max(alpha, v)
            if depth == self.depth:
                values[action] = v
            if alpha >= beta:
                break
            
        self.memory[s] = (depth, alpha)
        if depth == self.depth:
            return values
        return alpha

class AlphaMinimaxPlayer:
    def __init__(self, make_game, net, depth=4, n_sims=100, c_puct=1, show_pred=False, prepare_state=False):
        self.alpha_player = AlphaPlayer(make_game, net, n_sims=n_sims, c_puct=c_puct)
        self.minimax_player = MinimaxPlayer(make_game, depth)
        self.show_pred = show_pred
        self.prepare_state = prepare_state
    
    def play(self, state):
        if self.prepare_state:
            state = prepare(state)
        action = self.minimax_player.play(state)
        values_cache = self.minimax_player.values_cache
        print('Minimax value:', values_cache.max())
        if values_cache.max() <= 0:
            print('Minimax cannot find a solution, switch to AlphaZero')
            self.alpha_player.play(state)
            action_prob = self.alpha_player.action_prob
            if values_cache.max() == 0:
                action_prob *= (values_cache == 0)
            if action_prob.sum() == 0:
                print('AlphaZero cannot find a solution, switch back to Minimax')
                return action
            action = np.argmax(action_prob)
            if self.show_pred:
                print('AlphaZero value:', self.alpha_player.net.predict(state)[1])
                show_pi(action_prob)
            
        return action
    
def prepare(state):
    board = state[0].astype(np.int8)
    wboard = state[1].astype(np.int8)
    parts = state[2].diagonal()
    workers = np.concatenate([[np.where(state[1] == i)] for i in [-1, -2, 1, 2]])[:, :, 0]
    current_player = -1
    return (board, wboard, parts, workers, current_player)

def show_pi(pi):
    pi = np.reshape(np.round(pi, decimals=2), (2, 8, 8))
    dir_list = ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c']

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    sns.heatmap(pi[0], annot=True, annot_kws={'size': 10}, square=True, cbar=False, ax=ax[0], vmin=pi.min(), vmax=pi.max())
    ax[0].set_xticks(np.arange(8) + 0.5)
    ax[0].set_xticklabels(dir_list, rotation=0)
    ax[0].set_yticks(np.arange(8) + 0.5)
    ax[0].set_yticklabels(dir_list, rotation=0)

    sns.heatmap(pi[1], annot=True, annot_kws={'size': 10}, square=True, cbar=False, ax=ax[1], vmin=pi.min(), vmax=pi.max())
    ax[1].set_xticks(np.arange(8) + 0.5)
    ax[1].set_xticklabels(dir_list, rotation=0)
    ax[1].set_yticks(np.arange(8) + 0.5)
    ax[1].set_yticklabels(dir_list, rotation=0)
    plt.show()