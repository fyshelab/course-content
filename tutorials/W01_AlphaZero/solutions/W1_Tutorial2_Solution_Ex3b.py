class MCTSAgent(PolicyAgentBase):
    def __init__(self, tree:MCTSTree, temperature=1.0):
        super(MCTSAgent, self).__init__(temperature=temperature)
        self.tree = tree # you'll need this!
    
    def new_game(self):
        """Reset the tree for new games.
        """
        self.tree.init_tree()

    def get_policy(self, game:OthelloGame) -> torch.Tensor:
        """Input: game, an instance of OthelloGame
                  MCTStree, a built tree with the root as the starting board
         Returns: policy a (n_games, n, n) Tensor. The distribution over the 
                  board of "good" moves for the current player. 
        """
        # Run searches
        self.tree.run_searches(game)

        # The policy is proportional to visit counts of the children
        s = self.tree.state_key(game, game.current_player)
        policy = torch.zeros_like(game.boards)
        children = self.tree.Ls[s]
        for k, (i,j) in enumerate(children):
            policy[0,i,j] = self.tree.Nsa[s][k]
        return policy

# uncomment to test code
game = OthelloGame()
tree = MCTSTree(random_policy_value_fun, num_search=50)
my_mcts_agent = MCTSAgent(tree)
act = my_mcts_agent.select_move(game)
game.step(act)
game.render()
