from MCTS import MCTS
import numpy as np
import config
from Game import Game


class Player:
    def __init__(self, nn, budget, selfplay=False):
        self.nn = nn
        self.uct = 1
        self.budget = budget
        self.selfplay = selfplay
        self.tree = MCTS("NN", nn) if nn else MCTS("Random")
        self.current_node = None
        self.game = Game()
        if self.nn:
            self.nn.eval()

    def update_current_node(self, state):
        if self.current_node is None:
            self.current_node = self.tree.createNode(state)
        else:
            if len(self.current_node.children) == 0:
                self.tree.expand_all(self.current_node)
            self.current_node = next((node for node in self.current_node.children if node.state == state), None)
        self.current_node.parent = None

    def get_move(self, state):
        self.update_current_node(state)

        for sims in range(0, self.budget):
            self.tree.simulate(self.current_node, self.uct)

        visits = []
        for child in self.current_node.children:
            visits.append(child.N)

        max_visits = np.random.choice(np.where(visits == np.max(visits))[0])
        self.current_node = self.current_node.children[max_visits]

        return self.game.convert_move_to_col_index(self.current_node.move), []
