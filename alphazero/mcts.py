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
        self.Es = {}

    def get_action_prob(self, state, temp=1):
        s = self.game.tostring(state)
        
        for _ in range(self.n_sims):
            self.game.set_state(state)
            self.search(state, s, root=True)
        
        Nsa = self.Nsa
        counts = [Nsa[(s, a)] if (s, a) in Nsa else 0 for a in range(self.game.action_size)]
        
        if temp == 0:
            best_action = np.argmax(counts)
            probs = np.zeros(self.game.action_size)
            probs[best_action] = 1
        else:
            counts = np.array([c ** (1. / temp) for c in counts])
            probs = counts / float(sum(counts))

        return probs

    def search(self, state, s, root=False):
        if s in self.Es and self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ns:
            self.Qs[s] = 0
            self.Ns[s] = 0
            if s not in self.Vs:
                self.Vs[s] = self.game.legal_moves()
                self.Es[s] = self.game._reward_cache
                if self.Es[s] != 0:
                    return -self.Es[s]
            pi, v = self.net.predict(state)
            mask = np.zeros(self.game.action_size)
            mask[self.Vs[s]] = 1
            pi *= mask
            if sum(pi) > 0:
                pi /= sum(pi)
            else:
                print('Warning: predict non-legal moves')
                pi = mask
                pi /= sum(pi)
            self.Ps[s] = pi
            return -v
        
        valid_moves = self.Vs[s]
        best_action = -1
        best_qu = -np.inf
        
        epsilon = 0
        noise = np.zeros(self.game.action_size)
        if root:
            epsilon = self.epsilon
            noise[valid_moves] = np.random.dirichlet([self.alpha] * len(valid_moves))

        Ps = self.Ps[s]
        Qsa = self.Qsa
        Ns = self.Ns[s]
        Nsa = self.Nsa
        c_puct = self.c_puct
        
        for a in valid_moves:
            p = (1 - epsilon) * Ps[a] + epsilon * noise[a]
            if (s, a) in Qsa:
                q = Qsa[(s, a)]
                u = c_puct * p * (Ns ** 0.5) / (1 + Nsa[(s, a)])
            else:
                q = 0
                u = c_puct * p * ((Ns + 1e-8) ** 0.5)
            if q + u > best_qu:
                best_qu = q + u
                best_action = a
                
        a = best_action
        state, reward, done, legals = self.game.step(a)
        self.Es[s] = self.game._reward_cache
        if done:
            v = reward
        elif self.Es[s] != 0:
            v = self.Es[s]
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