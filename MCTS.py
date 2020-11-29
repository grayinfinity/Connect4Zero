import numpy as np
from Game import Game
import random
import config


class Node:
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


class MCTS:

    # ---------------------------------------------------------------------------- #
    def __init__(self, mcts_type, neural_net=None, use_dirichlet=True):
        self.neural_net = neural_net
        self.use_dirichlet = use_dirichlet
        self.type = mcts_type

    # ---------------------------------------------------------------------------- #
    def createNode(self, state, move=None, parent=None):
        node = Node(state, move, parent)
        return node

    # ---------------------------------------------------------------------------- #
    def PUCT(self, child, cpuct):
        game = Game()
        col_of_child = game.convert_move_to_col_index(child.move)
        return child.Q + cpuct * child.parent.proba_children[col_of_child] * np.sqrt(child.parent.N) / (1 + child.N)

    def UCT(self, node, c_uct):
        if node.N == 0:
            return 1000
        else:
            return node.Q + c_uct * np.sqrt(2 * np.log(node.parent.N) / (node.N))

    # ---------------------------------------------------------------------------- #
    def selection(self, node, c_uct):
        evaluator = self.PUCT if self.type == 'NN' else self.UCT
        if node.isLeaf():
            return node, node.isterminal()
        else:
            current = node
            while not current.isLeaf():
                values = np.asarray([evaluator(node, c_uct) for node in current.children])
                posmax = np.where(values == np.max(values))[0]
                current = current.children[np.random.choice(posmax)]
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
        game = Game(leaf.state)
        if not leaf.isterminal():
            flat = game.state_flattener(leaf.state)
            reward, P = self.neural_net.forward(flat)
            proba_children = P.detach().numpy()[0]
            NN_q_value = reward.detach().numpy()[0][0]

            if self.use_dirichlet and leaf.parent is None:
                probs = np.copy(proba_children)
                alpha = config.alpha_dir
                epsilon = config.epsilon_dir

                dirichlet_input = [alpha for _ in range(config.L)]
                dirichlet_list = np.random.dirichlet(dirichlet_input)
                proba_children = (1 - epsilon) * probs + epsilon * dirichlet_list

            leaf.W -= NN_q_value
            leaf.N += 1
            leaf.Q = leaf.W / leaf.N

            leaf.proba_children = proba_children

        else:
            _, winner = game.gameover()

            leaf.W += np.abs(winner)
            leaf.N += 1
            leaf.Q = leaf.W / leaf.N

    def random_rollout(self, node):
        gameloc = Game(node.state)
        if not node.isterminal():
            gameover = 0
            while not gameover:
                move = np.random.choice(gameloc.allowed_moves())
                gameloc.takestep(move)
                gameover, _ = gameloc.gameover()

        _, winner = gameloc.gameover()

        node.W += winner * node.parent.state[2]
        node.N += 1
        node.Q = node.W / node.N

    # ---------------------------------------------------------------------------- #
    def backFill(self, leaf):
        current = leaf
        count = 1
        reward = leaf.Q
        while current.parent is not None:
            current.parent.N += 1
            current.parent.W += ((-1) ** count) * reward
            current.parent.Q = current.parent.W / current.parent.N
            current = current.parent
            count += 1

    # ---------------------------------------------------------------------------- #
    def simulate(self, node, cpuct):
        leaf, isleafterminal = self.selection(node, cpuct)
        if not isleafterminal:
            self.expand_all(leaf)
            leaf = np.random.choice(leaf.children)
        if self.type == 'NN':
            self.eval_leaf(leaf)
        else:
            self.random_rollout(leaf)

        self.backFill(leaf)

    # ---------------------------------------------------------------------------- #
