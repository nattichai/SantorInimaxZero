import numpy as np

class MCTS:
    def __init__(self, game, net, n_sims, c_puct, epsilon=0, alpha=1):
        self.game = game
        self.net = net
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.epsilon = epsilon
        self.alpha = alpha

        self.Qsa = {}
        self.Nsa = {}
        self.Qs = {}
        self.Ns = {}
        self.Ps = {}
        self.Vs = {}

    def get_action_prob(self, state, temp=1):
        s = self.game.tostring(state)
        
        for _ in range(self.n_sims):
            self.game.set_state(state)
            self.search(state, s, root=True)
            
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.action_size)]
        
        if temp == 0:
            best_action = np.argmax(counts)
            probs = np.zeros(self.game.action_size)
            probs[best_action] = 1
        else:
            counts = np.array([c ** (1. / temp) for c in counts])
            probs = counts / float(sum(counts))

        return probs

    def search(self, state, s, root=False):
        if s not in self.Ns:
            self.Qs[s] = 0
            self.Ns[s] = 0
            if s not in self.Vs:
                self.Vs[s] = self.game.legal_moves()
            self.Ps[s], v = self.net.predict(state)
            mask = np.zeros(self.game.action_size)
            mask[self.Vs[s]] = 1
            self.Ps[s] *= mask
            if sum(self.Ps[s]) > 0:
                self.Ps[s] /= sum(self.Ps[s])
            else:
                print('Warning: predict non-legal moves')
                self.Ps[s] += mask
                self.Ps[s] /= sum(self.Ps[s])
            return -v
        
        valid_moves = self.Vs[s]
        best_action = -1
        best_qu = -np.inf
        
        epsilon = 0
        noise = np.zeros(self.game.action_size)
        if root:
            epsilon = self.epsilon
            noise[valid_moves] = np.random.dirichlet([self.alpha] * len(valid_moves))
                         
        for a in valid_moves:
            p = (1 - epsilon) * self.Ps[s][a] + epsilon * noise[a]
            if (s, a) in self.Qsa:
                q = self.Qsa[(s, a)]
                u = self.c_puct * p * (self.Ns[s] ** 0.5) / (1 + self.Nsa[(s, a)])
            else:
                q = 0
                u = self.c_puct * p * ((self.Ns[s] + 1e-8) ** 0.5)
            if q + u > best_qu:
                best_qu = q + u
                best_action = a
                
        a = best_action
        state, reward, done, legals = self.game.step(a)
        if done:
            v = reward
        else:
            ss = self.game.tostring(state)
            self.Vs[ss] = legals
            v = self.search(state, ss)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Qsa[(s, a)] * self.Nsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
            
        self.Qs[s] = (self.Qs[s] * self.Ns[s] + v) / (self.Ns[s] + 1)
        self.Ns[s] += 1

        return -v