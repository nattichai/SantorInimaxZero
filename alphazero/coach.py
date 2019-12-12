import os
import gzip
import numpy as np
from pickle import Pickler, Unpickler
from tqdm import tqdm_notebook
from random import shuffle
from .santorini import Santorini
from .mcts import MCTS

class Coach:
    def __init__(self, game, net, args):
        self.game = game
        self.net = net
        self.args = args
        
        self.mcts = None
        self.memories = []
        self.skip_first_selfplay = False
    
    def learn(self):
        for i in range(self.args.start_itr, self.args.n_itr + 1):
            print('------ITER ' + str(i) + '------')

            if not self.skip_first_selfplay or i > self.args.start_itr:
                self.memories.append([])
                for _ in tqdm_notebook(range(self.args.n_ep)):
                    self.run_episode()
            
            if self.args.slow_window:
                limit = min(self.args.base_history_itr + max(0, i - self.args.base_history_itr) // self.args.freq_history_itr, self.args.max_history_itr)
            else:
                limit = self.args.max_history_itr
            while len(self.memories) > limit:
                print(f'memories size exceed {limit} iters, remove oldest memory')
                self.memories.pop(0)
            
            self.save_memories(i - 1)
            
            train_data = []
            for memory in self.memories:
                train_data.extend(memory)
            
            if self.args.deduplicate:
                unique_train_data = {}
                for state, pi, v in train_data:
                    s = self.game.tostring(state)
                    if s not in unique_train_data:
                        unique_train_data[s] = [state, pi.copy(), v, 1]
                    else:
                        unique_train_data[s][1] += pi
                        unique_train_data[s][2] += v
                        unique_train_data[s][3] += 1
                train_data = [(state, pi / n, v / n) for state, pi, v, n in unique_train_data.values()]
            
            shuffle(train_data)
            self.net.train(train_data)
            self.net.save(self.get_model_file(i))

    def self_play(self, itr):
        while True:
            print('------ITER ' + str(itr) + '------')
            model_path = os.path.join(self.args.load_folder, 'model_' + str(itr - 1).zfill(4) + '.h5')
            self.net.load(model_path)
            print('Loaded', model_path)

            self.memories = [[]]
            for _ in tqdm_notebook(range(self.args.n_ep)):
                self.run_episode()
                self.save_memories(itr - 1)
                if os.path.isfile(os.path.join(self.args.load_folder, 'model_' + str(itr).zfill(4) + '.h5')):
                    break
            itr += 1
                    
    def run_episode(self):
        memory = []
        self.mcts = MCTS(Santorini(superpower=self.args.superpower,
                                   winning_floor=self.args.winning_floor,
                                   force_move=self.args.force_move),
                         self.net,
                         self.args.n_sims,
                         self.args.c_puct,
                         self.args.epsilon,
                         self.args.alpha)
        state = self.game.reset()
        s = self.game.tostring(state)
        step = 0
        while True:
            step += 1
            pi = self.mcts.get_action_prob(state, temp=int(step < self.args.no_temp_step))
            
            q = self.mcts.Qs[s]
            for st, p in self.game.get_symmetries(state, pi, self.args.sym):
                memory.append((st, self.game.current_player, p, q))
                
            action = np.random.choice(len(pi), p=pi)
            state, reward, done, legals = self.game.step(action)
            s = self.game.tostring(state)
            self.mcts.Vs[s] = legals
            
            if done:
                if self.args.avg_zq:
                    self.memories[-1] += [(state, pi, (reward * (-1) ** (player == self.game.current_player) + q) / 2) for state, player, pi, q in memory]
                else:
                    self.memories[-1] += [(state, pi, reward * (-1) ** (player == self.game.current_player)) for state, player, pi, q in memory]
                return
        
    def save_memories(self, i):
        if not os.path.exists(self.args.checkpoint):
            os.makedirs(self.args.checkpoint)
        try:
            with gzip.open(self.get_memory_file(i), "wb+", compresslevel=3) as f:
                Pickler(f).dump(self.memories)
        except:
            pass
    
    def get_model_file(self, i):
        return os.path.join(self.args.checkpoint, 'model_' + str(i).zfill(4) + '.h5')
    
    def get_memory_file(self, i):
        return os.path.join(self.args.checkpoint, 'memories_' + str(i).zfill(4))
    
    def load_memories(self):
        try:
            self.net.load(self.args.load_model_file)
        except:
            pass
        try:
            with gzip.open(self.args.load_memories_file, "rb") as f:
                self.memories = Unpickler(f).load()
            self.skip_first_selfplay = True
        except:
            pass