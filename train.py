from alphazero.santorini import Santorini
from alphazero.santorininet import SantoriniNet
from alphazero.coach import Coach

class EnvArgs:
    superpower: bool = True
    winning_floor: int = 3
    force_move: bool = True
        
class CoachArgs:
    start_itr: int = 1
    n_itr: int = 1000

    avg_zq: bool = True
    deduplicate: bool = False
    slow_window: bool = True

    n_ep: int = 100
    n_sims: int = 50
    c_puct: float = 3
    epsilon: float = 0.25
    alpha: float = 0.5
    
    base_history_itr: int = 4
    freq_history_itr: int = 2
    max_history_itr: int = 20
    no_temp_step: int = 18
    
    checkpoint: str = './checkpoint'
    load_model_file: str = checkpoint + '/model_0001.h5'
    load_memories_file: str = checkpoint + '/memories_0001'
        
    superpower: bool = EnvArgs.superpower
    winning_floor: int = EnvArgs.winning_floor
    force_move: bool = EnvArgs.force_move
        
class NetArgs:
    cyclic_lr: bool = True
    has_val: bool = True
    val_size: float = 0.2
    val_decay: float = 0.01

    n_ep: int = 8
    n_bs: int = 512
    base_lr: float = 1e-4
    max_lr: float = 1e-3
    
    n_denses: int = [512, 1024]
    dropout: float = 0.5
    n_ch: int = 256
    
game = Santorini(superpower=EnvArgs.superpower,
                 winning_floor=EnvArgs.winning_floor,
                 force_move=EnvArgs.force_move)
net = SantoriniNet(game, NetArgs)
coach = Coach(game, net, CoachArgs)
# coach.load_memories()
coach.learn()