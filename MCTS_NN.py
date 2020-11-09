#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             MCTS_NN.py
# Description:      Simulations in an MCTS guided by the NN player
# Authors:          Jean-Philippe Bruneton & Adele Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #


# ================================= PREAMBLE ================================= #
# Packages
import numpy as np
from Game_bitboard import Game
import random
import config


# ============================================================================ #

# =============================== CLASS: NODE ================================ #
# A class representing a node of a mcts
class Node:
    # ---------------------------------------------------------------------------- #
    # Constructs a node from a state
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move  # is the move (an int) that was played from parent to get there
        self.parent = parent
        self.children = []
        self.proba_children = np.zeros(config.L)

        self.N = 0  # vists
        self.W = 0  # cumulative reward
        self.Q = 0  # average reward

    def isLeaf(self):
        return len(self.children) == 0

    def isterminal(self):
        game = Game(self.state)
        gameover, _ = game.gameover()  # is 0 or 1
        return gameover


# ============================================================================ #


# =============================== CLASS: MCTS ================================ #
# A class representing a mcts
class MCTS_NN:

    # ---------------------------------------------------------------------------- #
    def __init__(self, player, use_dirichlet):
        self.root = None
        self.player = player
        self.use_dirichlet = use_dirichlet

    # ---------------------------------------------------------------------------- #
    def createNode(self, state, move=None, parent=None):
        node = Node(state, move, parent)
        return node

    # ---------------------------------------------------------------------------- #
    def PUCT(self, child, cpuct):
        game = Game()
        col_of_child = game.convert_move_to_col_index(child.move)
        return child.Q + cpuct * child.parent.proba_children[col_of_child] * np.sqrt(child.parent.N) / (1 + child.N)

    # ---------------------------------------------------------------------------- #
    def selection(self, node, cpuct):

        random.seed()

        # if the provided node is already a leaf (it shall happen only at the first sim)
        if node.isLeaf():
            return node, node.isterminal()

        else:  # the given node is not a leaf, thus pick a leaf descending from the node, according to PUCT :
            current = node

            while not current.isLeaf():
                values = []

                for child in current.children:
                    values += [self.PUCT(child, cpuct)]

                max_val = max(values)
                where_max = [i for i, j in enumerate(values) if j == max_val]

                if len(where_max) == 1:
                    current = current.children[where_max[0]]
                else:
                    imax = where_max[int(random.random() * len(where_max))]
                    current = current.children[imax]

        return current, current.isterminal()

    # ---------------------------------------------------------------------------- #
    def expand_all(self, leaf):

        game = Game(leaf.state)
        allowedmoves = game.allowed_moves()
        for move in allowedmoves:
            child = self.createNode(game.nextstate(move), move, parent=leaf)
            leaf.children += [child]

    # ---------------------------------------------------------------------------- #
    def eval_leaf(self, leaf):

        self.player.eval()
        np.random.seed()

        if not leaf.isterminal():

            game = Game(leaf.state)
            flat = game.state_flattener(leaf.state)

            # NN call
            reward, P = self.player.forward(flat)
            proba_children = P.detach().numpy()[0]
            NN_q_value = reward.detach().numpy()[0][0]

            if self.use_dirichlet and leaf.parent is None:
                probs = np.copy(proba_children)
                alpha = config.alpha_dir
                epsilon = config.epsilon_dir

                dirichlet_input = [alpha for _ in range(config.L)]
                dirichlet_list = np.random.dirichlet(dirichlet_input)
                proba_children = (1 - epsilon) * probs + epsilon * dirichlet_list

            leaf.W += NN_q_value
            leaf.N += 1
            leaf.Q = leaf.W / leaf.N

            leaf.proba_children = proba_children

        else:
            # seems reasonable to use the true value and not NN value
            game = Game(leaf.state)
            _, winner = game.gameover()

            if winner != 0:
                winner = 1 if winner == leaf.state[2] else -1

            leaf.W += winner
            leaf.N += 1
            leaf.Q = leaf.W / leaf.N

    # ---------------------------------------------------------------------------- #
    def backFill(self, leaf):

        current = leaf
        add_W = leaf.Q
        count = 1

        while current.parent is not None:
            current.parent.N += 1
            current.parent.W += ((-1) ** count) * add_W
            current.parent.Q = current.parent.W / current.parent.N
            # move up
            current = current.parent
            count += 1

    # ---------------------------------------------------------------------------- #
    def simulate(self, node, cpuct):

        leaf, isleafterminal = self.selection(node, cpuct)

        if not isleafterminal:
            self.expand_all(leaf)

        self.eval_leaf(leaf)

        self.backFill(leaf)

    # ---------------------------------------------------------------------------- #
