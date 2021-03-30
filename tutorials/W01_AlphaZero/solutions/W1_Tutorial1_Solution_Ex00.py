def count_pieces(game:OthelloGame, whoami):
    """A function to sum the number of pieces on the board of a given player
    current player (whoami).
    Inputs: game, an instance of OthelloGame
            whoami (int) 1 or 2, whose turn it is
    Outputs:
            populous_row: a 1d pytorch tensor of shape (n_games)"""

    n_my_pieces = torch.sum(game.boards==whoami, dim=(1,2))

    return n_my_pieces

game = OthelloGame(n_games=2) 
for turn in range(4): # take some actions in both games
    actions = actions_as_tensor(game.get_available_actions())
    act = actions[[0,1],[0,1]]
    game.step(act)

game.render()
print(count_pieces(game, 1))
