from MCTS import MCTS
import numpy as np
import config
from Game import Game


class Player:
    def __init__(self, nn, budget, selfplay=False):
        self.nn = nn
        self.uct = config.CPUCT
        self.budget = budget
        self.selfplay = selfplay
        self.tree = MCTS("nn", nn, self.selfplay) if nn else MCTS("Random")
        self.current_node = None
        self.game = Game()
        self.nn.eval()

    def update_current_node(self, state):
        if self.current_node is None:
            self.current_node = self.tree.createNode(state)
        else:
            self.current_node = next((node for node in self.current_node.children if node.state == state), None)

    def get_move(self, state, turn):
        self.update_current_node(state)

        for sims in range(0, self.budget):
            self.tree.simulate(self.current_node, self.uct)

        visits, childmoves = [], []
        for child in self.current_node.children:
            visits.append(child.N ** (1 / config.tau))
            childmoves.append(child.move)

        visit_probability = visits / np.sum(visits)

        if self.selfplay:
            child_col = np.asarray([self.game.convert_move_to_col_index(move) for move in childmoves], dtype=int)
            unmask_pi = np.zeros(config.L)
            unmask_pi[child_col] = visit_probability  # set all impossible moves to 0
            flatten_state = self.game.state_flattener(self.current_node.state)
            turn_data = np.hstack((flatten_state, unmask_pi, 0))
        else:
            turn_data = []

        if self.selfplay and turn < config.tau_zero_self_play:
            self.current_node = np.random.choice(self.current_node.children, p=visit_probability)
        else:
            max_visits = np.random.choice(np.where(visits == np.max(visits))[0])
            self.current_node = self.current_node.children[max_visits]

        return self.current_node.move, turn_data
