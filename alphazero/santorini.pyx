import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

cdef class Santorini:
    cdef public tuple board_dim
    cdef public starting_parts
    cdef public int winning_floor, superpower, n_win_dome
    cdef public force_move
    
    cdef public int current_player
    cdef public dict players
    cdef public board, workers, parts
    
    cdef public list moves, builds, itoa
    cdef public dict ktoc, atoi
    cdef public int action_size
    
    def __init__(self,
                 tuple board_dim=(5, 5),
                 starting_parts=np.asarray([0, 22, 18, 14, 18]),
                 int winning_floor=3,
                 int superpower=True,
                 int n_win_dome=5,
                 force_move=False):
        self.board_dim = board_dim
        self.starting_parts = starting_parts
        self.winning_floor = winning_floor
        self.superpower = superpower
        self.n_win_dome = n_win_dome
        self.force_move = force_move
        
        self.moves = ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c']
        self.builds = ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c']
        self.ktoc = {
            'q': np.asarray((-1, -1)),
            'w': np.asarray((-1, 0)),
            'e': np.asarray((-1, 1)),
            'a': np.asarray((0, -1)),
            'd': np.asarray((0, 1)),
            'z': np.asarray((1, -1)),
            'x': np.asarray((1, 0)),
            'c': np.asarray((1, 1))
        }
        self.itoa = [(w, m, b) for w in [1, 2] for m in self.moves for b in self.builds]
        self.atoi = {a: i for i, a in enumerate(self.itoa)}
        self.action_size = 128
        
        self.reset()
        
    def reset(self):
        self.current_player = -1
        self.board = np.zeros(self.board_dim, np.int8)
        self.workers = np.zeros(self.board_dim, np.int8)
        self.parts = self.starting_parts.copy()
        self.players = {-1: np.asarray([(0, 2), (4, 2)]),
                        1: np.asarray([(2, 0), (2, 4)])}
        self.workers[0, 2], self.workers[4, 2] = -1, -2
        self.workers[2, 0], self.workers[2, 4] = 1, 2
        return self.get_state()
        
    def get_state(self):
        return self.board, -self.current_player * self.workers, self.parts, self.players, self.current_player
    
    def set_state(self, tuple state):
        self.board, self.workers, self.parts, self.players, self.current_player = state[0].copy(), state[1].copy() * -state[4], state[2].copy(), {-1: state[3][-1].copy(), 1: state[3][1].copy()}, state[4]
    
    def legal_moves(self, force_move=None):
        if force_move is None:
            force_move = self.force_move
        cdef list legals = []
        cdef int worker, i
        cdef str move, build
        cdef pos, move_pos, build_pos
        cdef dict build_dests = {}
        
        for worker in [1, 2]:
            pos = self.players[self.current_player][worker - 1]
            for move in self.moves:
                move_pos = pos + self.ktoc[move]
                if -1 < move_pos[0] < self.board_dim[0] and -1 < move_pos[1] < self.board_dim[1] and \
                 self.workers[move_pos[0], move_pos[1]] == 0 and \
                 self.board[move_pos[0], move_pos[1]] - self.board[pos[0], pos[1]] <= 1 and \
                 self.board[move_pos[0], move_pos[1]] < 4:
                    for build in self.builds:
                        build_pos = move_pos + self.ktoc[build]
                        if -1 < build_pos[0] < self.board_dim[0] and -1 < build_pos[1] < self.board_dim[1] and \
                         ((build_pos[0] == pos[0] and build_pos[1] == pos[1]) or \
                          self.workers[build_pos[0], build_pos[1]] == 0) and \
                          self.board[build_pos[0], build_pos[1]] <= 3:
                            i = self.atoi[(worker, move, build)]
                            if force_move:
                                if (self.board[move_pos[0], move_pos[1]] == self.winning_floor) or \
                                 (self.superpower and self.board[build_pos[0], build_pos[1]] == 3 and \
                                  self.parts[4] == self.starting_parts[4] - self.n_win_dome + 1):
                                    return [i]
                                if (build_pos[0], build_pos[1]) not in build_dests:
                                    build_dests[(build_pos[0], build_pos[1])] = [i]
                                else:
                                    build_dests[(build_pos[0], build_pos[1])].append(i)
                            legals.append(i)
        if force_move:
            for worker in [1, 2]:
                pos = self.players[-self.current_player][worker - 1]
                for move in self.moves:
                    move_pos = pos + self.ktoc[move]
                    if -1 < move_pos[0] < self.board_dim[0] and -1 < move_pos[1] < self.board_dim[1] and \
                     self.workers[move_pos[0], move_pos[1]] == 0 and \
                     self.board[move_pos[0], move_pos[1]] - self.board[pos[0], pos[1]] <= 1 and \
                     self.board[move_pos[0], move_pos[1]] < 4:
                        if (move_pos[0], move_pos[1]) in build_dests:
                            if self.board[move_pos[0], move_pos[1]] == self.winning_floor:
                                return build_dests[(move_pos[0], move_pos[1])]
                            if self.board[move_pos[0], move_pos[1]] + 1 == self.winning_floor and self.board[pos[0], pos[1]] == self.winning_floor - 1:
                                for i in build_dests[(move_pos[0], move_pos[1])]:
                                    legals.remove(i)
                            build_dests.pop((move_pos[0], move_pos[1]))
        return legals
    
    def step(self, int action):
        cdef int worker
        cdef str move, build
        cdef pos, move_pos, build_pos
        cdef int reward=0, done=0
        
        worker, move, build = self.itoa[action]
        pos = self.players[self.current_player][worker - 1]
        move_pos = pos + self.ktoc[move]
        build_pos = move_pos + self.ktoc[build]
        
        self.workers[pos[0], pos[1]] = 0
        self.players[self.current_player][worker - 1] = move_pos
        self.workers[move_pos[0], move_pos[1]] = self.current_player * worker
        
        if self.board[move_pos[0], move_pos[1]] == self.winning_floor:
            reward, done = 1, 1
        
        cdef int new_high = self.board[build_pos[0], build_pos[1]] + 1
        if self.parts[new_high] > 0:
            self.parts[new_high] -= 1
            self.board[build_pos[0], build_pos[1]] += 1
            if self.superpower and new_high == 4 and self.starting_parts[4] - self.parts[4] == self.n_win_dome:
                reward, done = 1, 1
            
        self.current_player *= -1
        cdef legals = self.legal_moves()
        if len(legals) == 0:
            reward, done = 1, 1
            
        return self.get_state(), reward, done, legals
    
    def tostring(self, tuple state):
        return state[0].tostring() + state[1].tostring()
    
    def get_symmetries(self, tuple state, pi, sym=False):
        if not sym:
            return [((state[0].copy(), state[1].copy(), state[2].copy(), {-1: state[3][-1].copy(), 1: state[3][1].copy()}, state[4]), pi)]
        dir_board = np.array([['q', 'w', 'e'],
                              ['a', ' ', 'd'],
                              ['z', 'x', 'c']])
        cdef list symmetries = [], env_workers
        for rot in range(4):
            new_buildings = np.rot90(state[0], rot)
            new_workers = np.rot90(state[1].copy(), rot)
            new_dir_board = np.rot90(dir_board, rot)
            
            for flip in [False, True]:
                if flip:
                    new_buildings = np.fliplr(new_buildings)
                    new_workers = np.fliplr(new_workers)
                    new_dir_board = np.fliplr(new_dir_board)
                    
                for swap_op in [False, True]:
                    if swap_op:
                        new_workers[state[3][1][0][0], state[3][1][0][1]], new_workers[state[3][1][1][0], state[3][1][1][1]] = 2, 1
                
                    for swap_cur in [False, True]:
                        env_workers = [1, 2]
                        if swap_cur:
                            new_workers[state[3][-1][0][0], state[3][-1][0][1]], new_workers[state[3][-1][1][0], state[3][-1][1][1]] = -2, -1
                            env_workers = [2, 1]
                            
                        new_pi = np.array([pi[self.atoi[(w, m, b)]] for w in env_workers \
                                                                          for m in new_dir_board.ravel() \
                                                                            for b in new_dir_board.ravel() \
                                                                              if m != ' ' and b != ' '])
                        symmetries.append(((new_buildings.copy(), new_workers.copy(), state[2].copy(), {-1: state[3][-1].copy(), 1: state[3][1].copy()}, state[4]), new_pi))
        return symmetries
    
    def display(self, state):
        board, workers, parts, players, turn = state
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(board, vmin=-0.5, vmax=4.5, cmap=plt.cm.get_cmap('Blues', 5), alpha=0.8)

        ax.text(players[-1][0][1], players[-1][0][0], -1, size=30, ha='center', va='center', color='r')
        ax.text(players[-1][1][1], players[-1][1][0], -2, size=30, ha='center', va='center', color='r')
        ax.text(players[1][0][1], players[1][0][0], 1, size=30, ha='center', va='center', color='g')
        ax.text(players[1][1][1], players[1][1][0], 2, size=30, ha='center', va='center', color='g')

        cbar = plt.colorbar(im, ax=ax, ticks=range(5), shrink=0.8)
        cbar.set_ticklabels([f'{i} floor: {parts[i]} parts' for i in range(5)])
        cbar.ax.tick_params(labelsize=15)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(6) - 0.5)
        ax.set_yticks(np.arange(6) - 0.5)
        ax.grid(color='k', linewidth=1)
        plt.title(f'Turn: {turn}')
        plt.show()