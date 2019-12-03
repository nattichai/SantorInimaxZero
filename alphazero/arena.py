class Arena():
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        players = [self.player1, None, self.player2]
        state = self.game.reset()
        while True:
            if verbose:
                self.display(state)
            action = players[self.game.current_player + 1](state)
            state, reward, done, _ = self.game.step(action)
            if done:
                return reward * self.game.current_player
    
    def play_games(self, num, verbose=False):
        num = int(num / 2)
        p1wins, p2wins = 0, 0
        for _ in range(num):
            result = self.play_game(verbose=verbose)
            print(1 if result == 1 else 2, end='')
            if result == 1:
                p1wins += 1
            else:
                p2wins += 1
        self.player1, self.player2 = self.player2, self.player1
        print(' ', end='')
        for _ in range(num):
            result = self.play_game(verbose=verbose)
            print(2 if result == 1 else 1, end='')
            if result == 1:
                p2wins += 1
            else:
                p1wins += 1
        print()
        return p1wins, p2wins