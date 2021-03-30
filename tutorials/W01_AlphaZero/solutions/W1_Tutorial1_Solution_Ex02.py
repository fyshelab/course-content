def my_board_value(board_states, whoami=1):
    """
    Returns the *value* of a board or set of boards.
    
    The value is a number between -1 and 1 that indicates how good that board is
    for the player. Specifically, the value is a measure of confidence
    that a board state will lead to a final win.
    1 is an estimated win, -1 a loss, and 0 means a draw.

    Inputs: board_states (torch tensor with size [n_games, n, n]):
                0s are positions with no piece:
                1 is player 1, 
                2 is player 2.
            whoami is an integer (1 or 2) indicating which player "you" are.
    Return +1 if the "whoami" player is sure to win, -1 if the "other" player is
    sure to win, 0 for a draw, or any value in between.

    Outputs: torch array (1d tensor) of values, one per n_games 

    Note: this function must not modify board_states!
    """
    n_games, n, _ = board_states.size()
    
    values = torch.zeros(n_games)
    n1 = (board_states == OthelloGame.PLAYER1).view(n_games, -1).sum(dim=1)
    n2 = (board_states == OthelloGame.PLAYER2).view(n_games, -1).sum(dim=1)

    frac1 = n1/(n1+n2)
    values = frac1*2-1

    return values if whoami == 1 else -values

game = OthelloGame()
game.step(game.get_available_actions()[0][0])
game.step(game.get_available_actions()[0][1])
game.step(game.get_available_actions()[0][0])
game.render()

print(f"Value for player 1 (dark) is {my_board_value(game.boards, whoami=1)}")
print(f"Value for player 2 (light) is {my_board_value(game.boards, whoami=2)}")