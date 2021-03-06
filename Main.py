import numpy as np
from Game import Game
import random
from MCTS import MCTS
from Player import Player
import config
from ResNet import ResNet_Training
import os
import ResNet
import torch
import ray
import copy
from tqdm import tqdm
import json

ray.init(num_gpus=4)


def main():
    nn_budget = 100
    improved = 0
    best_nn = load_or_create_neural_net()
    best_nn.to(torch.device("cuda"))
    elos = []
    last_elo = 30
    data = []

    for i in range(1000):
        new_data = self_play(best_nn, nn_budget + improved)

        previous_best = copy.deepcopy(best_nn)
        improve_model_resnet(best_nn, new_data, improved)

        print('Compering the new NN vs the old one')
        winrate = test_player(best_nn, nn_budget, previous_best, nn_budget, 10)
        print("Winrate: " + str(winrate))
        if winrate < 0.52:
            best_nn = previous_best
        else:
            improved += 1
            #  last_elo = check_elo(best_nn, nn_budget, last_elo)
            #  elos.append(last_elo)
            torch.save(best_nn.state_dict(), './best_model_resnet.pth')
            with open('elos.txt', 'w') as filehandle:
                json.dump(elos, filehandle)

    print(elos)


def test_player(main_player_nn, main_player_budget, compare_player_nn, compare_player_budget, loops):
    games_count = 0
    main_player_wins = 0
    main_player_nn_id = ray.put(main_player_nn)
    compare_player_nn_id = ray.put(compare_player_nn)
    for i in range(loops):
        result_ids = []
        for _ in range(config.CPUS):
            if i < loops // 2:  # check if making a copy is enough
                result_ids.append(
                    [play_game.remote(main_player_nn_id, main_player_budget, compare_player_nn_id,
                                      compare_player_budget, False), 1])
            else:
                result_ids.append(
                    [play_game.remote(compare_player_nn_id, compare_player_budget, main_player_nn_id,
                                      main_player_budget, False), -1])

        for result_id, player_number in result_ids:
            winner, data = ray.get(result_id)
            if winner == player_number:
                main_player_wins += 1
            if winner != 0:
                games_count += 1

    del main_player_nn_id
    del compare_player_nn_id

    winrate = main_player_wins / games_count

    return winrate


def check_elo(nn, budget, start_elo):
    mcts_budget = start_elo
    print('Playing vs MCTS')
    for _ in range(10):
        winrate = test_player(nn, budget, False, mcts_budget, 2)
        if winrate > 0.5:
            mcts_budget += 10
        else:
            mcts_budget = max(0, mcts_budget - 10)
            if mcts_budget == 0:
                break

    print("Elo: " + str(mcts_budget))
    return mcts_budget


def self_play(nn, budget):
    print('Self playing')

    cpus = config.CPUS
    if not nn:
        cpus = 96
    winners = []
    all_data = []
    nn_id = ray.put(nn)
    for _ in tqdm(range(30)):
        result_ids = []
        for i in range(cpus):
            result_ids.append(play_game.remote(nn_id, budget, nn_id, budget, True))

        for result_id in result_ids:
            winner, data = ray.get(result_id)
            winners.append(winner)
            if len(all_data) == 0:
                all_data = data
            else:
                all_data = np.vstack((all_data, data))
    del nn_id

    return all_data


@ray.remote(num_gpus=0.2)
def play_game_with_gpu(player_one_nn, player_one_budget, player_two_nn, player_two_budget, is_selfplay):
    return play_game(player_one_nn, player_one_budget, player_two_nn, player_two_budget, is_selfplay)


@ray.remote
def play_game_no_gpu(player_one_nn, player_one_budget, player_two_nn, player_two_budget, is_selfplay):
    return play_game(player_one_nn, player_one_budget, player_two_nn, player_two_budget, is_selfplay)


@ray.remote(num_gpus=0.2)
def play_game(player_one_nn, player_one_budget, player_two_nn, player_two_budget, is_selfplay):
    player_one = Player(player_one_nn, player_one_budget, is_selfplay)
    player_two = Player(player_two_nn, player_two_budget, is_selfplay)
    gameover, winner, turn = False, 0, 0
    game = Game()
    new_data_for_the_game = np.zeros((3 * config.L * config.H + config.L + 1))

    while not gameover:
        turn += 1
        if turn % 2 == 1:
            move, turn_data = player_one.get_move(game.state, turn)
        else:
            move, turn_data = player_two.get_move(game.state, turn)

        if is_selfplay:
            new_data_for_the_game = np.vstack((new_data_for_the_game, turn_data))

        game.takestep(move)
        gameover, winner = game.gameover()

    if is_selfplay:
        new_data_for_the_game = np.delete(new_data_for_the_game, 0, 0)
        history_size = new_data_for_the_game.shape[0]

        z = 0
        if winner == 1:
            z = 1 - config.long_game_factor * history_size / 42  # the reward is bigger for shorter games
        elif winner == -1:
            z = -1 + config.long_game_factor * history_size / 42  # the reward is less negative for long games

        z_vec = np.zeros(history_size)

        for i in range(history_size):
            z_vec[i] = ((-1) ** i) * z

        new_data_for_the_game[:, -1] = z_vec
        new_data_for_the_game = extend_data(new_data_for_the_game)

    return winner, new_data_for_the_game


def extend_data(data):
    board_size = config.L * config.H
    extended_data = np.zeros((data.shape[0], data.shape[1]))

    for i in range(extended_data.shape[0]):
        board = np.copy(data[i, 0:3 * board_size]).reshape((3, config.H, config.L))
        yellowboard = board[0]
        redboard = board[1]
        player_turn = board[2]

        # parity operation on array for both yellow and red boards
        flip_yellow = np.fliplr(yellowboard)
        flip_red = np.fliplr(redboard)

        extended_data[i, 0:board_size] = flip_yellow.flatten()
        extended_data[i, board_size:2 * board_size] = flip_red.flatten()
        extended_data[i, 2 * board_size: 3 * board_size] = player_turn.flatten()

        # parity operation on the Pi's
        pi_s = np.copy(data[i, 3 * board_size:3 * board_size + config.L])
        flip_pi = np.flip(pi_s, axis=0)

        extended_data[i, 3 * board_size:3 * board_size + config.L] = flip_pi
        extended_data[i, -1] = np.copy(data[i, -1])

    return np.vstack((data, extended_data))


def load_or_create_neural_net():
    file_path = './best_model_resnet.pth'
    if os.path.exists(file_path):
        print('loading already trained model')
        best_player_so_far = ResNet.resnet18()
        best_player_so_far.load_state_dict(torch.load(file_path))

    else:
        print('Trained model doesnt exist. Starting from scratch.')
        best_player_so_far = ResNet.resnet18()

    best_player_so_far.eval()
    return best_player_so_far


# ---------------------------------------------------------------------------- #
# Neural Net training
def improve_model_resnet(nn, data, i):
    # here i is the number of times NN has improved : it will be used for learning rate annealing

    min_data = config.MINIBATCH * config.MINBATCHNUMBER
    max_data = config.MINIBATCH * config.MAXBATCHNUMBER
    size_training_set = data.shape[0]

    if size_training_set >= min_data:
        print('size of train set', size_training_set)

        # random sample from past experiences :
        if size_training_set >= max_data:
            X = data[np.random.choice(data.shape[0], max_data, replace=False)]
            # X = data[-size_training_set:-1,:]
        else:
            X = data

        # learning rate annealing: decrease by config.lrdecay the learning rate every config.annealing succesfull improvements of the model
        j = i // config.annealing

        if config.optim == 'sgd':
            lr_decay = config.sgd_lr / ((config.lrdecay) ** j)
        elif config.optim == 'adam':
            lr_decay = config.adam_lr / ((config.lrdecay) ** j)

        k = (i - 1) // config.annealing
        if j == k + 1:
            print('learning rate is now = ', lr_decay)

        training = ResNet_Training(nn, config.MINIBATCH, config.EPOCHS, lr_decay, X, X, 1)
        training.trainNet()

    else:
        print('Not enough training data. Please increase number of self play games or use previous data = True')
        print('train set size', data.shape[0])
        raise ValueError


def load_or_create_neural_net():
    file_path = './best_model_resnet.pth'
    best_player_so_far = ResNet.resnet18()
    if os.path.exists(file_path):
        print('loading already trained model')
        best_player_so_far.load_state_dict(torch.load(file_path))

    best_player_so_far.eval()

    return best_player_so_far


if __name__ == "__main__":
    main()
