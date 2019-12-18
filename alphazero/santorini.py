import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Santorini:
    def __init__(
            self,
            board_dim=(5, 5),
            starting_parts=np.array([0, 22, 18, 14, 18]),
            winning_floor: int = 3,
            auto_invert: bool = True,
            superpower: bool = False,
            n_win_dome: int = 5,
            force_move: bool = False,
            prepare_state: bool = False):
        self.board_dim = board_dim
        self.starting_parts = starting_parts
        self.winning_floor = winning_floor
        self.auto_invert = auto_invert
        self.superpower = superpower
        self.n_win_dome = n_win_dome
        self.force_move = force_move
        self.prepare_state = prepare_state

        self.moves = ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c']
        self.builds = ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c']

        self.ktoc = np.asarray([(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1] if x != 0 or y != 0])
        self.itoa = [(w, m, b) for w in [1, 2] for m in self.moves for b in self.builds]
        self.atoi = {action: index for index, action in enumerate(self.itoa)}
        self.wtoi = {-1: 0, -2: 1, 1: 2, 2: 3}
        self.action_size = 128
        self._legal_move_cache = None
        self._reward_cache = None

        self.reset()

    def get_state(self):
        w_board = np.zeros(self.board_dim, dtype=np.int8)
        w_board[self._workers[0][0], self._workers[0][1]] = -1
        w_board[self._workers[1][0], self._workers[1][1]] = -2
        w_board[self._workers[2][0], self._workers[2][1]] = 1
        w_board[self._workers[3][0], self._workers[3][1]] = 2

        if self.auto_invert:
            w_board *= -self.current_player
        return self._board, w_board, self._parts, self._workers, self.current_player

    def set_state(self, state):
        self._done = False
        self._legal_move_cache = None
        self._reward_cache = None
        self._board, self._workers, self._parts, self.current_player = state[0].copy(), state[3].copy(), state[2].copy(), state[4]

    def reset(self):
        self.current_player = -1
        self._board = np.zeros(self.board_dim, dtype=np.int8)
        self._workers = np.array([
            (0, 2),
            (4, 2),
            (2, 0),
            (2, 4),
        ])
        self._parts = self.starting_parts.copy()
        self._done = False
        self._legal_move_cache = None
        self._reward_cache = None
        return self.get_state()

    def legal_moves(self):
        assert not self._done, "must reset"
        if self._legal_move_cache is not None:
            return self._legal_move_cache
        
        self._legal_move_cache, self._reward_cache = _legal_moves(self._workers, self._board, self._parts, self.current_player, self.winning_floor, self.superpower, self.force_move, self.starting_parts, self.n_win_dome)
        return self._legal_move_cache

    def step(self, action):
        assert not self._done, "must reset"
        worker, mdir, bdir = int(action/64), int((action%64)/8), action%8

        wid = worker if self.current_player < 0 else worker + 2
        mdir = self.ktoc[mdir]
        bdir = self.ktoc[bdir]
        
        walkable, moved, is_win = _walkable(wid, mdir, self._workers, self._board, self.winning_floor)
        if walkable:
            reward, done = 0., False
            self._workers[wid] = moved
            
            if is_win:
                reward, done = 1., True
            else:
                buildable, part, built = _buildable(moved, bdir, wid, self._workers, self._board, self._parts)
                if buildable:
                    self._board[built[0], built[1]] = part
                    self._parts[part] -= 1
                    if self.superpower:
                        n_dome = (self._board == 4).sum()
                        if n_dome == self.n_win_dome:
                            reward, done = 1., True
                else:
                    raise ValueError('illegal move')
        else:
            raise ValueError('illegal move')

        self.current_player *= -1
        self._legal_move_cache = None
        self._reward_cache = None
        legals = self.legal_moves()
        
        if len(legals) == 0:
            reward, done = 1., True

        self._done = done
        return self.get_state(), reward, done, legals
    
    def tostring(self, state):
        return state[0].tostring() + state[1].tostring()
    
    def get_symmetries(self, state, pi, sym=False):
        board, w_board, parts, workers, current_player = state[0].copy(), state[1].copy(), state[2].copy(), state[3].copy(), state[4]
        if not sym:
            return [((board, w_board, parts, workers, current_player), pi)]
        dir_board = np.array([['q', 'w', 'e'],
                              ['a', ' ', 'd'],
                              ['z', 'x', 'c']])
        symmetries = []
        for rot in range(4):
            new_buildings = np.rot90(board, rot)
            new_workers = np.rot90(w_board, rot)
            new_dir_board = np.rot90(dir_board, rot)
            
            for flip in [False, True]:
                if flip:
                    new_buildings = np.fliplr(new_buildings)
                    new_workers = np.fliplr(new_workers)
                    new_dir_board = np.fliplr(new_dir_board)
                    
                for swap_op in [False, True]:
                    if swap_op:
                        new_workers[workers[2][0], workers[2][1]], new_workers[workers[3][0], workers[3][1]] = 2, 1
                
                    for swap_cur in [False, True]:
                        env_workers = [1, 2]
                        if swap_cur:
                            env_workers = [2, 1]
                            new_workers[workers[0][0], workers[0][1]], new_workers[workers[1][0], workers[1][1]] = -2, -1
                            
                        new_pi = np.array([pi[self.atoi[(w, m, b)]] for w in env_workers \
                                                                        for m in new_dir_board.ravel() \
                                                                            for b in new_dir_board.ravel() \
                                                                                if m != ' ' and b != ' '])
                        symmetries.append(((new_buildings.copy(), new_workers.copy(), parts, workers, current_player), new_pi))
        return symmetries

    def display(self, state):
        if self.prepare_state:
            state = self.prepare(state)
        board, _, parts, workers, current_player = state
        _, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(board, vmin=-0.5, vmax=4.5, cmap=plt.cm.get_cmap('Blues', 5), alpha=0.8)

        ax.text(workers[0][1], workers[0][0], -1, size=30, ha='center', va='center', color='r')
        ax.text(workers[1][1], workers[1][0], -2, size=30, ha='center', va='center', color='r')
        ax.text(workers[2][1], workers[2][0], 1, size=30, ha='center', va='center', color='g')
        ax.text(workers[3][1], workers[3][0], 2, size=30, ha='center', va='center', color='g')

        cbar = plt.colorbar(im, ax=ax, ticks=range(5), shrink=0.8)
        cbar.set_ticklabels([f'{i} floor: {parts[i]} parts' for i in range(4)] + [f'Dome: {parts[4]} parts'])
        cbar.ax.tick_params(labelsize=13)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(6) - 0.5)
        ax.set_yticks(np.arange(6) - 0.5)
        ax.grid(color='k', linewidth=1)
        plt.title(f'Current player: {current_player}')
        plt.show()
        
    def prepare(self, state):
        board = state[0].astype(np.int8)
        wboard = state[1].astype(np.int8)
        parts = state[2].diagonal()
        workers = np.concatenate([[np.where(state[1] == i)] for i in [-1, -2, 1, 2]])[:, :, 0]
        current_player = -1
        return (board, wboard, parts, workers, current_player)

@njit
def _walkable(
        wid: int,
        dir: np.ndarray,
        workers: np.ndarray,
        board: np.ndarray,
        winning_floor: int
):
    # check boundary
    src = workers[wid]
    new = src + dir
    board_dim = board.shape
    if not (0 <= new[0] < board_dim[0]): return False, new, False
    if not (0 <= new[1] < board_dim[1]): return False, new, False

    # not a dome
    tgt = board[new[0], new[1]]
    if tgt == 4: return False, new, False

    # not too high
    cur = board[src[0], src[1]]
    if tgt > cur + 1: return False, new, False

    # no other worker
    for i in range(len(workers)):
        if workers[i][0] == new[0] and workers[i][1] == new[1]:
            return False, new, False

    return True, new, board[new[0], new[1]] == winning_floor

@njit
def _buildable(
        src: np.ndarray,
        dir: np.ndarray,
        wid: int,
        workers: np.ndarray,
        board: np.ndarray,
        parts: np.ndarray,
):
    # check boundary
    new = src + dir
    board_dim = board.shape
    if not (0 <= new[0] < board_dim[0]): return False, wid, new
    if not (0 <= new[1] < board_dim[1]): return False, wid, new

    # not a dome
    tgt = board[new[0], new[1]]
    if tgt == 4: return False, wid, new

    # no other worker
    for i in range(len(workers)):
        if i != wid:
            if workers[i][0] == new[0] and workers[i][1] == new[1]:
                return False, wid, new

    # check parts
    if parts[tgt + 1] == 0: return False, wid, new

    return True, tgt + 1, new

@njit()
def _legal_moves(workers: np.ndarray,
                 board: np.ndarray,
                 parts: np.ndarray,
                 current_player: int,
                 winning_floor: int,
                 superpower: bool,
                 force_move: bool,
                 starting_parts: np.ndarray,
                 n_win_dome: int):
    legals = []
    ktoc = np.asarray([(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1] if x != 0 or y != 0])
    if force_move:
        w_board = np.zeros((5, 5))
        w_board[workers[0][0], workers[0][1]] = -1
        w_board[workers[1][0], workers[1][1]] = -2
        w_board[workers[2][0], workers[2][1]] = 1
        w_board[workers[3][0], workers[3][1]] = 2
        
        al1 = workers[2 * (current_player > 0)]
        al2 = workers[2 * (current_player > 0) + 1]
        op1 = workers[2 * (current_player < 0)]
        op2 = workers[2 * (current_player < 0) + 1]
        blocks = None
        prepare = False
        winnings = []

    for worker in range(2):
        src = al1 if worker == 0 else al2
        al = al2 if worker == 0 else al1
        for move in range(8):
            moved = src + ktoc[move]
            high = board[moved[0], moved[1]]
            if 0 <= moved[0] and moved[0] < 5 and 0 <= moved[1] and moved[1] < 5 and \
                w_board[moved[0], moved[1]] == 0 and high - board[src[0], src[1]] <= 1 and high != 4:
                for build in range(8):
                    action = 64 * worker + 8 * move + build
                    if high == winning_floor:
                        if force_move:
                            return [action], 1 # Instant win
                        legals.append(action)
                        continue
                    built = moved + ktoc[build]
                    part = board[built[0], built[1]] + 1
                    if 0 <= built[0] and built[0] < 5 and 0 <= built[1] and built[1] < 5 and \
                        (w_board[built[0], built[1]] == 0 or (built[0] == src[0] and built[1] == src[1])) and part <= 4 and parts[part] > 0:
                        if force_move:
                            if superpower and part == 4 and starting_parts[4] - n_win_dome + 1 == parts[4]:
                                return [action], 1 # Instant win
                            if part == winning_floor and ((board[op1[0], op1[1]] == winning_floor - 1 and max(np.abs(op1 - built)) == 1) or \
                                (board[op2[0], op2[1]] == winning_floor - 1 and max(np.abs(op2 - built)) == 1)):
                                continue # Never give a win
                            if blocks is None:
                                blocks = [[-1] for _ in range(25)]
                            blocks[built[0] * 5 + built[1]].append(action) # Prepare instant blocking
                            legals.append(action)

                            if not prepare:
                                prepare = True
                                fars = np.zeros((5, 5), dtype=np.bool_)
                                for i in range(5):
                                    for j in range(5):
                                        if (abs(i - op1[0]) > 2 or abs(j - op1[1]) > 2) and \
                                            (abs(i - op2[0]) > 2 or abs(j - op2[1]) > 2):
                                            fars[i, j] = 1
                                
                                thirds = np.zeros((5, 5), dtype=np.int8)
                                forks = np.zeros((5, 5), dtype=np.int8)
                                for i in range(5):
                                    for j in range(5):
                                        if board[i][j] != winning_floor - 1 or w_board[i, j] != 0:
                                            continue
                                        cur = np.array([i, j])
                                        cnt2 = 0
                                        cnt3 = 0
                                        for dir in range(8):
                                            near = cur + ktoc[dir]
                                            if near[0] < 0 or near[0] >= 5 or near[1] < 0 or near[1] >= 5 or w_board[near[0], near[1]] != 0:
                                                continue
                                            if board[near[0], near[1]] == winning_floor - 1:
                                                cnt2 = dir + 1
                                            elif board[near[0], near[1]] == winning_floor:
                                                if cnt3 == 0:
                                                    cnt3 = dir + 1
                                                else:
                                                    cnt3 += 10 * (dir + 1)
                                        if cnt3 != 0:
                                            thirds[i, j] = cnt3
                                        if cnt2 != 0 and cnt3 != 0:
                                            forks[i, j] = cnt2 # Prepare fork winning
                                        elif cnt3 >= 10:
                                            forks[i, j] = -cnt3 # Prepare fork winning

                            if high == winning_floor - 1:
                                fork = forks[moved[0], moved[1]]
                                if fork != 0:
                                    if fork > 0:
                                        if build == fork - 1:
                                            winnings.append(action) # Fork to win
                                    elif build != (-fork // 10) - 1 and build != (-fork % 10) - 1:
                                        winnings.append(action) # Fork to win
                                else:
                                    third = thirds[moved[0], moved[1]]
                                    if third != 0 and build != third - 1:
                                        d = ktoc[third - 1]
                                        if fars[moved[0] + d[0]][moved[1] + d[1]] == 1:
                                            winnings.append(action) # Faraway winning

                            if part == winning_floor and (high == winning_floor - 1 or (board[al[0], al[1]] == winning_floor - 1 and max(np.abs(al - built)) == 1)):
                                if fars[built[0], built[1]] == 1:
                                    winnings.append(action) # Faraway winning
                        else:
                            legals.append(action)
    if force_move:
        legals_cop = legals.copy()
        for worker in range(2):
            src = op1 if worker == 0 else op2
            for move in range(8):
                moved = src + ktoc[move]
                high = board[moved[0], moved[1]]
                if 0 <= moved[0] and moved[0] < 5 and 0 <= moved[1] and moved[1] < 5 and \
                    w_board[moved[0], moved[1]] == 0 and high - board[src[0], src[1]] <= 1 and high != 4:
                    if high == winning_floor:
                        if len(blocks[moved[0] * 5 + moved[1]]) > 1:
                            return blocks[moved[0] * 5 + moved[1]][1:], 0 # Instant blocking
                        else:
                            return legals_cop, -1 # Cannot blocking
        
        if len(winnings) > 0:
            return winnings, 1 # Fork to win or Faraway winning
            
        if len(legals) == 0:
            return legals_cop, -1 # No move left
    return legals, 0